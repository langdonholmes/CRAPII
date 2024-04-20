# CRAPII
The Cleaned Repository of Annotated Personally Identifiable Information (CRAPII) is a dataset of 22,000+ student essays labeled for PII. The original identifiers have been replaced with surrogate identifiers to protect the identity of the student authors. The dataset is intended for use in training and evaluating machine learning models for PII detection.

This repository contains several code artifacts that describe the most important steps in the data cleaning and PII obfuscation process. The intent is to provide a reproducible process that may be applied to other datasets. Although we cannot share the original data, we hope that the code may be useful to others.

## Code

Code artifacts are presented in three iPython notebooks and two Python scripts.

- `src/data_cleaing.ipynb` - This notebook describes the process of cleaning the original dataset. It includes processes for removing near-duplicate documents. It also demonstrates a method for removing documents that were likely to be corrupted using anomaly detection based on character distributions.
- `src/manual_obfuscate.ipynb` - This notebook describes the process of manually obfuscating PII labeled as "Other". It includes an iPython widget for viewing labeled text and manually typing out an alternative (non-identifying) surrogate.
- `src/obfuscate.py` - This Python script uses different methods to generate plausible surrogates for each type of PII (except the "Other" type). Several methods rely on `Faker`, and all methods are orchestrated using `Presidio`.
- `src/obfuscate.ipynb` - This notebook runs the `obfuscate.py` script and exports the data into several formats.
- `load_json_examply.py` is a short script that demonstrates how to load the obfuscated JSON Lines data into a Pandas DataFrame.

## Data

The data can be downloaded from Kaggle: [CRAPII](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data).

Additional details about the data format are available on the Kaggle website.

# Citation

If you use this dataset in your research, please cite the following paper:

Holmes, L., Crossley, S. A., Wang, J., & Zhang, W. (2024). Cleaned Repostiory of Annotated PII. Proceedings of the 17th International Conference on Educational Data Mining (EDM).



