# House Price Prediction Dashboard

[Open the completed Colab notebook](https://colab.research.google.com/drive/1Y7HLCtqWNCWhQnaB4MGjNsv0c_dURC1V).
This new notebook is saved in My Drive. The original notebook was preserved
because the connected Drive app was not granted permission to overwrite it.

Open the updated Colab notebook and run its new workflow from top to bottom.
If `01_House_Price.xlsx` is already in `/content`, it is used automatically.
Otherwise, the notebook asks you to upload it. The workbook must contain a
sheet named `Data`. No example or synthetic data is substituted for your file.

The notebook trains a new model, writes `app.py`, and downloads a deployment ZIP.
It preserves the original exploratory cells in a clearly marked appendix.
Training can take several minutes on a Colab CPU.

## Files to deploy

```text
house_price_dashboard/
├── app.py
├── house_price_model.pkl       # Created after training on your workbook
├── 01_House_Price.xlsx         # Optional for prediction; required for price bands
├── requirements.txt           # Pinned to the training environment automatically
├── .streamlit/config.toml
├── train.py                   # Optional on Streamlit Community Cloud
├── model_metadata.json        # Human-readable audit; also stored inside the model
└── model_comparison.csv       # Training cross-validation results
```

For a local deployment, use Python 3.11 or 3.12 and run:

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

If you are starting with the source files instead of the trained Colab ZIP:

```sh
python -m pip install -r requirements.txt
python train.py --data 01_House_Price.xlsx --output .
streamlit run app.py
```

On Streamlit Community Cloud, upload the deployment files to your repository,
select `app.py` as the entrypoint, and use the generated pinned requirements.
Keep the Python version compatible with the notebook's training environment.
The dashboard also checks `/content/` as a fallback for model/data files.
Only load the model artifact created by your own trusted notebook.

## Training and interpretation

- Five-fold cross-validation compares a mean-price baseline, Ridge regression,
  Random Forest, Extra Trees, and histogram gradient boosting, including a small
  hyperparameter grid. The lowest mean validation RMSE determines the winner.
- A seeded 80/20 train/test split reserves the test set until selection is done.
  The saved model remains fitted to the training partition; it is not subsequently
  refitted on the test data. The displayed scores belong to that saved model.
- Numeric missing values use training-fold medians and StandardScaler. Missing
  categories use the training-fold mode. One-hot encoding keeps every category.
  All transformations are fitted within each training fold through a Pipeline.
- `record_id`, `house_price_rm`, and unrelated columns never become predictors.
- The seven canonical inputs are `floor_area_m2`, `bedrooms`, `bathrooms`,
  `property_age_years`, `location_zone`, `property_type`, and `parking_spaces`.
  Explicit aliases are documented in `train.py`; an ambiguous schema stops training.
- Rows missing the target and exact duplicate feature-plus-target rows are removed.
  Missing input values are retained for imputation. Invalid numeric values and
  unexpected category labels stop training with an explanatory error.
- R² is not an accuracy percentage. RMSE and MAE use RM units. The best model among
  these candidates is not guaranteed to be the best possible model or to predict
  unseen properties accurately.
- The split assumes rows are independent properties. Repeated observations of a
  property, spatial clustering, or time ordering require a group/time-aware split
  before using the evaluation for deployment decisions.
- The target column `house_price_rm` is the currency contract. Prices and errors
  are displayed as Ringgit Malaysia. The app makes no time-period assumption.
- The dataset's target quartiles define relative price bands, not official tiers.
  Scenario differences are model associations, not causal effects or appraisals.
- Prediction timestamps use explicitly labelled UTC. Results persist during the
  current Streamlit session; a browser reload can start a new session.

## Verification status

The full workflow has been executed on the supplied `01_House_Price.xlsx` found
in `Downloads/IFJC4002_All_10_Project_Datasets`. The workbook has 2,000 rows and
the same 16 missing floor areas and 16 missing bedroom counts recorded in the
original notebook. The Guide sheet explicitly identifies it as synthetic
teaching data; there are no real property transactions in this dataset.

Selected model: **HistGradientBoostingRegressor**, chosen by training-only five-fold
cross-validation across 26 configurations. The saved artifact was evaluated on
400 held-out rows, with 1,600 rows used for training.

| Held-out metric | Result |
|---|---:|
| R² | 0.9486 |
| RMSE | RM 91,911.65 |
| MAE | RM 65,065.87 |
| Mean-price baseline RMSE | RM 405,313.58 |

These measured scores describe the supplied teaching workbook, not real-world
valuation accuracy. The trained model is included in the ready-to-run ZIP.
Package versions and the workbook SHA-256 are recorded in model_metadata.json.
Separate integration tests used generated fixtures to check errors, missing files,
imputation, scenario comparisons, and persistent results. Those fixture scores
are not included as evaluation results.

## Documentation

- [scikit-learn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Streamlit caching and state](https://docs.streamlit.io/develop/api-reference/caching-and-state)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)

Developed by Ir. Dr. Marni Azira Markom, Faculty of Intelligent Computing,
Universiti Malaysia Perlis (UniMAP).
