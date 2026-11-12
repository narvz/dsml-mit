# Data

The datasets are **not included in this repository**. Most of them were provided by the MIT ADSP course (Great Learning) and aren't mine to redistribute.
Every notebook keeps its saved outputs, so all charts and results are visible on GitHub without the data. To **re-run** a notebook, place the files below in this folder.

| File | Used in | Source |
|---|---|---|
| `customer_personality.csv` (tab-separated) | Clustering, preprocessing, dimensionality reduction | Public: Kaggle, *Customer Personality Analysis* (`marketing_campaign.csv`, renamed) |
| `superkart.csv` | Regression, Python/pandas, statistics | MIT ADSP: Module 3 Guided Project (`SuperKart-2.csv`, renamed) |
| `extraalearn.csv` | Classification, deep learning, preprocessing, statistics | MIT ADSP: Module 4 Project Assessment (`ExtraaLearn.csv`, renamed) |
| `shinkansen/traveldata_train.csv.gz` | Shinkansen project | MIT ADSP Hackathon: `Traveldata_train.csv`, gzipped |
| `shinkansen/surveydata_train.csv.gz` | Shinkansen project | MIT ADSP Hackathon: `Surveydata_train.csv`, gzipped |
| `shinkansen/traveldata_test.csv.gz` | Shinkansen project | MIT ADSP Hackathon: `Traveldata_test.csv`, gzipped |
| `shinkansen/surveydata_test.csv.gz` | Shinkansen project | MIT ADSP Hackathon: `Surveydata_test.csv`, gzipped |

Downloaded automatically, nothing to place here:
- **Fashion-MNIST** (deep learning): `keras.datasets.fashion_mnist`
- **MovieLens 100k** (recommendation systems): `surprise`
- **Digits** (dimensionality reduction): `sklearn.datasets.load_digits`

To gzip a CSV: `python -c "import pandas as pd; pd.read_csv('Traveldata_train.csv').to_csv('traveldata_train.csv.gz', index=False)"`
