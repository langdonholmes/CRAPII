import logging
from typing import Optional, Literal
from collections import defaultdict
import random
import string
import re
import json
from urllib.parse import urlparse
import pandas as pd

from nameparser import HumanName
from names_dataset import NameDataset
from faker import Faker
from spacy.tokens import Doc

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, RecognizerResult
from presidio_anonymizer.operators import OperatorType

logger = logging.getLogger("obfuscator")


class NameGetter(NameDataset):
    """A wrapper around the names_dataset.NameDataset class.
    NameDataset uses some data structures that are optimized for finding
    the most likely gender/country of a name.
    """

    def __init__(self) -> None:
        super().__init__()

    def _max(self, name_stats: dict) -> str:
        try:
            g_ = {a: b for a, b in name_stats.items() if b is not None}
            return max(g_, key=g_.get)
        except ValueError:
            return None

    def get_gender(self, name: HumanName) -> str:
        """Return the most frequent gender code for the provided name's
        first name, or None if a match cannot be found.
        """
        result = self.first_names.get(name.first)
        return self._max(result.get("gender", None)) if result else None

    def get_country(self, name: HumanName) -> str:
        """Return the most frequent country code for the provided name's
        last name, or None if a match cannot be found.
        """
        result = self.last_names.get(name.last)
        return self._max(result.get("country", None)) if result else None


class SurrogateAnonymizer(AnonymizerEngine):
    """A class for replacing human names with plausible surrogates.

    Constants:
        name_table: path to a parquet file containing with columns:
        ['first', 'last', 'gender', 'country']

        obfuscation_dict: path to a json file structured like this:
        Dict[PII_TYPE[PII_VALUE]] = [REPLACEMENT]

    Args:
        AnonymizerEngine (presidio_anonymizer.AnonymizerEngine):
        A Presidio anonymizer engine
    """

    names_df_path = "data/ascii_names.parquet"
    obfuscation_map_path = "data/other_mapping.json"
    pii_path = "data/pii.json"
    date_digits = re.compile(r"(?:\d{4}|\d{1,2})")

    def __init__(self, remember_replacements: Literal["all", "document"] = "all"):
        """
        Args:
            remember_replacements (Literal["all", "document"], default="all"):
            Surrogate PII will be replaced with the same value every time it is
            seen within the configured scope.

            "all" will remember all replacements made during the session.
            This may be useful for related documents, such as a collection of
            discussion board posts.

            "document" will remember replacements made during the current document.
            This is optimal for unrelated documents, such as a collection of essays.
        """
        super().__init__()
        self.fake = Faker()

        self.name_getter = NameGetter()
        self.names_df = pd.read_parquet(self.names_df_path)

        with open(self.obfuscation_map_path, "r") as f:
            self.obfuscation_map = json.load(f)

        with open(self.pii_path, "r") as f:
            self.all_pii = json.load(f)

        # keep track of PII we see and the replacement we generate
        self.remember = remember_replacements
        self.seen_pii = defaultdict(dict)

    def set_seeds(self, doc: Doc) -> None:
        """Set the seed for random number generators.
        Uses the hash of the input text to ensure that the same seed is used
        for each document across runs (even if the order of the documents changes)."""
        self.seed = abs(hash(doc.text) % 2**32)
        Faker.seed(self.seed)
        random.seed(self.seed)

    def randomize_characters(self, pii: str) -> str:
        surrogate = ""
        # if there are no ASCII characters to randomize, return whitespace
        # of equal length to the original string
        if not any(c.isascii() for c in pii):
            return " " * len(pii)

        for c in pii:
            if c.isdigit():
                surrogate += random.choice(string.digits)
            elif c.islower():
                surrogate += random.choice(string.ascii_lowercase)
            elif c.isupper():
                surrogate += random.choice(string.ascii_uppercase)
            else:
                surrogate += c

        return surrogate

    def get_random_name(
        self, country: Optional[str] = None, gender: Optional[str] = None
    ) -> pd.Series:
        """Returns a random name from the database as pd.Series.
        Matches gender and country, if provided.
        :country: ISO country code e.g. 'CO' for Columbia
        :gender: 'M' or 'F'
        """

        names_view = self.names_df

        if country:
            names_view = names_view[names_view["country"] == country]

        if gender:
            names_view = names_view[names_view["gender"] == gender]

        if names_view.shape[0] < 50:
            # If we don't have enough names, just return a random sample
            # logger.info(f"Too few names matching {gender} + {country}.")
            return self.names_df.sample(
                n=1, weights=self.names_df["count"], random_state=self.seed
            )

        return names_view.sample(
            n=1, weights=names_view["count"], random_state=self.seed
        )

    def generate_surrogate_name(self, original_name: str) -> str:
        """Generate a surrogate name."""

        if original_name == "PII":
            # Every time we call this function, Presidio will validate it
            # by testing that the function returns a str when the input is
            # 'PII'. We don't need to run below code in this case.
            return "PII"

        # Use nameparser to split the name
        name = HumanName(original_name.strip().title())
        new_name = HumanName()
        gender, country = None, None

        # First check if we have seen this name before
        if name.last:
            if name.last in self.seen_pii["NAME_LAST"]:
                # logger.info("Last name has already been seen.")
                new_name.last = self.seen_pii["NAME_LAST"][name.last]
            else:
                # Sample last name, attempting to match country.
                country = self.name_getter.get_country(name)
                # logger.info(f"Country set to {country}")
                new_name.last = self.get_random_name(
                    country=country,
                )[
                    "last"
                ].iloc[0]
                # logger.info(f"Last name surrogate is {new_name.last}")

        if name.first:
            if name.first in self.seen_pii["NAME_FIRST"]:
                # logger.info("First name has already been seen.")
                new_name.first = self.seen_pii["NAME_FIRST"][name.first]
            else:
                # Sample first name, attempting to match gender and country.
                gender = self.name_getter.get_gender(name)
                # logger.info(f"Gender set to {gender}")
                new_name.first = self.get_random_name(
                    gender=gender,
                    country=country,
                )[
                    "first"
                ].iloc[0]
                # logger.info(f"First name surrogate is {new_name.first}")

        logger.info(
            f"{original_name} --> Gender: {gender}, Country: {country} --> {new_name}"
        )

        self.seen_pii["NAME_FIRST"][name.first] = new_name.first
        self.seen_pii["NAME_LAST"][name.last] = new_name.last

        return str(new_name)

    def _operator_log(self, pii: str, surrogate: str):
        if pii != "PII":
            logger.info(f"{pii} --> {surrogate}")

    def fake_url(self, pii: str) -> str:
        if pii not in self.seen_pii["URL"]:
            url = urlparse(pii)

            # for youtube.com, randomize the query after v=
            if re.match(r"(www\.)?youtube", url.netloc):
                if re.match(r"\/user\/", url.path):
                    new_path = "/user/" + self.fake.user_name()
                    url = url._replace(path=new_path)
                new_query = url.query[:2] + self.randomize_characters(url.query[2:])
                url = url._replace(query=new_query).geturl()

            # for youtu.be, randomize the path
            elif re.match(r"(www\.)?youtu.be", url.netloc):
                new_path = self.randomize_characters(url.path)
                url = url._replace(path=new_path).geturl()

            # for facebook/instagram, generate username for path
            # remove query just in case
            elif re.match(r"(www\.)?(facebook|insta)", url.netloc):
                new_path = self.fake.user_name()
                url = url._replace(path=new_path)._replace(query="").geturl()

            # for linkedin/com/in/, randomize the path after /in/
            # remove query just in case
            elif re.match(r"(www\.)?linkedin", url.netloc):
                new_path = url.path[:4] + self.fake.user_name()
                url = url._replace(path=new_path)._replace(query="").geturl()
                print(url)

            # if the URL is complex, generate a URL with a path
            elif url.path or url.query:
                url = self.fake.uri()

            # otherwise, generate a simple URL
            else:
                url = self.fake.url()

            self.seen_pii["URL"][pii] = url

        surrogate = self.seen_pii["URL"][pii]
        self._operator_log(pii, surrogate)
        return surrogate

    def fake_email(self, pii: str) -> str:
        if pii not in self.seen_pii["EMAIL"]:
            self.seen_pii["EMAIL"][pii] = self.fake.free_email()
        surrogate = self.seen_pii["EMAIL"][pii]
        self._operator_log(pii, surrogate)
        return surrogate

    def fake_phone(self, pii: str) -> str:
        if pii not in self.seen_pii["PHONE_NUM"]:
            self.seen_pii["PHONE_NUM"][pii] = self.fake.phone_number()
        surrogate = self.seen_pii["PHONE_NUM"][pii]
        self._operator_log(pii, surrogate)
        return surrogate

    def fake_address(self, pii: str) -> str:
        if pii not in self.seen_pii["STREET_ADDRESS"]:
            self.seen_pii["STREET_ADDRESS"][pii] = self.fake.address()
        surrogate = self.seen_pii["STREET_ADDRESS"][pii]
        self._operator_log(pii, surrogate)
        return surrogate

    def fake_username(self, pii: str) -> str:
        if pii not in self.seen_pii["USERNAME"]:
            self.seen_pii["USERNAME"][pii] = self.fake.user_name()
        surrogate = self.seen_pii["USERNAME"][pii]
        self._operator_log(pii, surrogate)
        return surrogate

    def fake_date(self, pii: str) -> str:
        """Generate a fake date using a simple regex pattern.
        Captures 1, 2 or 4 digits and replaces them with a random integer.
        Ensures that the replacement has the same string length as the original."""
        if pii not in self.seen_pii["DATE"]:
            surrogate = pii
            for match in self.date_digits.finditer(pii):
                start, end = match.span()
                date_num = match.group()
                if len(date_num) == 1:
                    new_num = str(random.randint(1, 9))
                elif len(date_num) == 2:
                    new_num = random.randint(1, 31)
                    new_num = str(new_num).zfill(2)
                else:
                    # sample from distribution centered on observed date minus 5.
                    # this makes future dates less likely.
                    # std dev of 3 to ensure that most dates are within 10 years.
                    new_num = round(random.gauss(int(date_num) - 5, 3))
                    new_num = str(new_num).zfill(len(date_num))

                surrogate = surrogate[:start] + new_num + surrogate[end:]

            self.seen_pii["DATE"][pii] = surrogate

        surrogate = self.seen_pii["DATE"][pii]
        self._operator_log(pii, surrogate)
        return surrogate

    def fake_user_id(self, pii: str) -> str:
        """Generate a fake ID by randomly mutating ascii characters"""
        if pii not in self.seen_pii["USER_ID"]:
            surrogate = self.randomize_characters(pii)
            self.seen_pii["USER_ID"][pii] = surrogate

        surrogate = self.seen_pii["USER_ID"][pii]
        self._operator_log(pii, surrogate)
        return surrogate

    def shuffle_location(self, pii: str) -> str:
        return self.shuffle_obfuscate(pii, "LOCATION")

    def shuffle_education(self, pii: str) -> str:
        return self.shuffle_obfuscate(pii, "EDUCATION")

    def shuffle_employer(self, pii: str) -> str:
        return self.shuffle_obfuscate(pii, "EMPLOYER")

    def shuffle_obfuscate(self, pii: str, pii_type: str) -> str:
        """Return a previously seen obfuscated value for the provided pii_type"""

        if pii not in self.seen_pii[pii_type]:
            self.seen_pii[pii_type][pii] = random.choice(self.all_pii[pii_type])

        surrogate = self.seen_pii[pii_type][pii]
        if pii != "PII":
            logger.info(f"{pii} --> {surrogate}")
        return surrogate

    def map_obfuscate(self, pii: str) -> str:
        surrogate = self.obfuscation_map.get(pii, "")

        if surrogate == "NOT_PII":
            surrogate = pii

        if pii != "PII":
            logger.info(f"{pii} --> {surrogate}")
        return surrogate

    def extract_ents(self, doc: Doc) -> list[RecognizerResult]:
        """Return a list of RecognizerResults using the .ents attribute of
        a Spacy Doc object."""

        results = []
        for ent in doc.ents:
            spacy_result = RecognizerResult(
                entity_type=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                score=1.0,
            )
            results.append(spacy_result)

        return results

    def anonymize(self, doc: Doc) -> str:
        """Anonymize identified input using Presidio Anonymizer."""

        if not doc:
            logging.warning("No input provided to anonymize.")
            return

        self.set_seeds(doc)

        if self.remember == "document":
            # reset seen_pii for each document
            self.seen_pii = defaultdict(dict)

        entity_data = self._remove_conflicts_and_get_text_manipulation_data(
            self.extract_ents(doc)
        )

        operators = self._AnonymizerEngine__check_or_add_default_operator(
            {
                "NAME_STUDENT": OperatorConfig(
                    "custom", {"lambda": self.generate_surrogate_name}
                ),
                "EMAIL": OperatorConfig("custom", {"lambda": self.fake_email}),
                "PHONE_NUM": OperatorConfig("custom", {"lambda": self.fake_phone}),
                "URL_PERSONAL": OperatorConfig("custom", {"lambda": self.fake_url}),
                "STREET_ADDRESS": OperatorConfig(
                    "custom", {"lambda": self.fake_address}
                ),
                "USERNAME": OperatorConfig("custom", {"lambda": self.fake_username}),
                "ID_NUM": OperatorConfig("custom", {"lambda": self.fake_user_id}),
                "AGE": OperatorConfig("keep", {}),
                "DATE": OperatorConfig("custom", {"lambda": self.fake_date}),
                "NAME_INSTRUCTOR": OperatorConfig(
                    "custom", {"lambda": self.generate_surrogate_name}
                ),
                "OTHER": OperatorConfig("custom", {"lambda": self.map_obfuscate}),
                "LOCATION": OperatorConfig("keep", {}),
                "EMPLOYER": OperatorConfig("custom", {"lambda": self.shuffle_employer}),
                "EDUCATION": OperatorConfig(
                    "custom", {"lambda": self.shuffle_education}
                ),
            }
        )

        res = self._operate(doc.text, entity_data, operators, OperatorType.Anonymize)

        return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    anonymizer = SurrogateAnonymizer()

    test_names = [
        "Nora Wang",
        "John Williams",
        "John H. Williams",
        "MJ",
        "",
        "(",
        "Mario Escobar Sanchez",
        "Jane Fonda Michelle Rousseau",
        "Sir Phillipe Ricardo de la Sota Mayor",
        "Anthony REDDIX",
        "REDDIX, Anthony",
    ]

    for test_name in test_names:
        name = HumanName(test_name.strip().title())
        logger.info(f"{test_name} -> {name.first} + {name.last}")

        anonymizer.generate_surrogate(test_name)
