# Data

The project uses the public **Credit Card Fraud Detection** dataset (ULB Machine Learning Group, published on Kaggle):
284,807 card transactions made by European cardholders over two days in September 2013, of which 492 are fraud.
Features `V1`-`V28` are PCA components (the original features are confidential), plus `Time`, `Amount` and `Class`.

Download `creditcard.csv` from <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> and place it here as
`data/creditcard.csv`. It is not committed to this repository because of its size and its own licence terms
(Database Contents License) - please read them on the dataset page.

Without the file, `python run_all.py --demo` runs the whole pipeline on a small synthetic stand-in with the same columns.
