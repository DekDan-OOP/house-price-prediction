"""Student-friendly house price dashboard. Run: streamlit run app.py.

Use the fitted Pipeline exported by the companion training notebook. The
dashboard NEVER fits an estimator, encoder, imputer, or scaler.
"""

from datetime import datetime, timezone
from pathlib import Path
import math
import warnings

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


FEATURES = ["floor_area_m2", "bedrooms", "bathrooms", "property_age_years",
            "location_zone", "property_type", "parking_spaces"]
NUMERIC = [name for name in FEATURES if name not in ("location_zone", "property_type")]
ZONES = ["Urban_Centre", "Suburban", "Town_Periphery", "Rural"]
TYPES = ["Apartment", "Terrace", "Semi_Detached", "Detached"]
LABELS = dict(zip(FEATURES, ["Floor Area (m²)", "Bedrooms", "Bathrooms",
                           "Property Age (Years)", "Location Zone",
                           "Property Type", "Parking Spaces"]))
DISCLAIMER = (
    "This dashboard is developed for teaching and learning. Predictions are "
    "estimates from a machine-learning model and do not constitute an official "
    "real estate valuation, bank appraisal, or financial quote."
)


def resolve_file(name):
    """Prefer deployment files beside app.py; fall back to Colab's /content."""
    for folder in (Path(__file__).resolve().parent, Path("/content")):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def fingerprint(path):
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@st.cache_resource
def load_model(path, modified_ns, size):
    """Only load the trusted model produced by your training notebook."""
    # Version mismatches can produce unreliable results: fail with a clear message.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise ValueError(
            "The saved object is a bare estimator, not a preprocessing-and-model "
            "Pipeline. Run the companion training notebook to export the complete "
            "fitted pipeline; feature names alone cannot recover preprocessing."
        )
    check_is_fitted(model)
    names = list(getattr(model, "feature_names_in_", []))
    if names != FEATURES:
        raise ValueError(f"Expected input columns in this order: {FEATURES}. Found: {names}.")
    metadata = getattr(model, "dashboard_metadata_", {})
    if (metadata.get("schema_version") != "house-price-v1"
            or metadata.get("target") != "house_price_rm"):
        raise ValueError(
            "This pipeline has no verified dashboard preprocessing/currency metadata. "
            "Use the complete pipeline exported by the companion training notebook."
        )
    if "preprocess" not in model.named_steps:
        raise ValueError("The saved pipeline is missing its fitted preprocessing step.")
    return model, metadata


@st.cache_data
def load_dataset(path, modified_ns, size):
    return pd.read_excel(path, sheet_name="Data")


def estimator_name(model):
    estimator = model
    while isinstance(estimator, Pipeline):
        estimator = estimator.steps[-1][1]
    return type(estimator).__name__


def prepare_inputs(records, model):
    """Validate raw columns; the saved pipeline performs every transformation."""
    raw = pd.DataFrame(records)
    if set(raw.columns) != set(FEATURES):
        raise ValueError("Input columns do not match the seven property features.")
    raw = raw.loc[:, list(model.feature_names_in_)].copy()
    for name in NUMERIC:
        raw[name] = pd.to_numeric(raw[name], errors="raise").astype(float)
        if not raw[name].map(math.isfinite).all():
            raise ValueError(f"{LABELS[name]} must be a finite number.")
    for name, categories in (("location_zone", ZONES), ("property_type", TYPES)):
        if not raw[name].isin(categories).all():
            raise ValueError(f"Unknown category in {name}.")
    if (raw["floor_area_m2"] <= 0).any() or (raw[NUMERIC[1:]] < 0).any().any():
        raise ValueError("Floor area must be positive; counts and age cannot be negative.")
    for name in ("bedrooms", "bathrooms", "parking_spaces"):
        if (raw[name] % 1 != 0).any():
            raise ValueError(f"{LABELS[name]} must be a whole number.")
    # Neither record_id nor house_price_rm is ever passed to the model.
    return raw


def predict_prices(model, raw):
    """Predict through the entire fitted pipeline, without any fitting."""
    values = [float(value) for value in model.predict(raw)]
    if len(values) != len(raw):
        raise ValueError("The model returned an unexpected number of predictions.")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(
            "The model returned an invalid price estimate (negative or non-finite). "
            "No replacement value has been substituted."
        )
    return values


def transformed_vector(model, raw):
    """Expose the actual final-estimator inputs for learning and inspection."""
    preprocessing = model[:-1]
    values = preprocessing.transform(raw)
    if hasattr(values, "toarray"):
        values = values.toarray()
    names = preprocessing.get_feature_names_out()
    return pd.DataFrame(values, columns=names, index=raw.index)


def valuation_thresholds(dataset):
    if dataset is None or "house_price_rm" not in dataset.columns:
        return None
    prices = pd.to_numeric(dataset["house_price_rm"], errors="coerce")
    prices = prices[prices.map(lambda value: pd.notna(value)
                              and math.isfinite(float(value)) and value >= 0)]
    if prices.empty:
        return None
    return tuple(float(value) for value in prices.quantile([0.25, 0.5, 0.75]))


def price_category(price, thresholds):
    if thresholds is None:
        return "Unavailable"
    for label, upper in zip(("Low", "Moderate", "High"), thresholds):
        if price <= upper:
            return label
    return "Very High"


def show_performance(metadata, name):
    st.write(f"**Model:** {name}")
    evaluation = metadata.get("held_out_test", {})
    # Scores come ONLY from the held-out evaluation saved with this exact model.
    for label, key in (("R²", "r2"), ("RMSE", "rmse"), ("MAE", "mae")):
        value = evaluation.get(key)
        valid = isinstance(value, (int, float)) and math.isfinite(value)
        display = (f"{value:.4f}" if key == "r2" else f"RM {value:,.2f}") if valid else "Not provided"
        st.write(f"**{label}:** {display}")
    if evaluation:
        st.caption(f"Held-out test rows: {evaluation.get('rows', 'Not provided')}. "
                   "Model selection used training-fold validation only.")
    st.caption("R² describes model fit and can be negative. RMSE and MAE measure "
               "prediction error in RM; lower is better on the same evaluation data.")


def show_preprocessing(model, metadata, raw=None, vector=None):
    with st.expander("How the inputs are prepared"):
        st.write("Seven raw property columns enter the saved pipeline. The target "
                 "house_price_rm and identifier record_id are excluded.")
        st.write("Numeric missing values use training-fold medians, followed by "
                 "standard scaling. Categories use the training-fold most frequent "
                 "value for missing values, then one-hot encoding. No category is "
                 "dropped, so there is no dummy-variable reference category.")
        preprocessor = model.named_steps["preprocess"]
        numeric = preprocessor.named_transformers_["numeric"]
        categorical = preprocessor.named_transformers_["categorical"]
        st.dataframe(pd.DataFrame({
            "Feature": NUMERIC,
            "Fitted median": numeric.named_steps["impute"].statistics_,
            "Fitted scaling mean": numeric.named_steps["scale"].mean_,
            "Fitted scaling divisor": numeric.named_steps["scale"].scale_,
        }), hide_index=True, use_container_width=True)
        encoder = categorical.named_steps["encode"]
        mappings = []
        for feature, categories, fill in zip(
                ["location_zone", "property_type"], encoder.categories_,
                categorical.named_steps["impute"].statistics_):
            for category in categories:
                mappings.append({"Feature": feature, "Category": str(category),
                                 "Encoded column": f"categorical__{feature}_{category}",
                                 "Value": "1 when matched; otherwise 0",
                                 "Missing-value fill": str(fill)})
        st.dataframe(pd.DataFrame(mappings), hide_index=True, use_container_width=True)
        if metadata.get("column_aliases"):
            st.write("Workbook column names mapped during training:")
            st.json(metadata["column_aliases"])
        st.caption("Statistics shown are fitted on the training partition. The "
                   "dashboard never learns preprocessing from the Excel file or form.")
        if raw is not None:
            st.write("**Submitted raw input row**")
            st.dataframe(raw, hide_index=True, use_container_width=True)
        if vector is not None:
            st.write("**Actual transformed feature vector passed to the estimator**")
            st.dataframe(vector, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
    st.markdown("""<style>
    .stApp { background: #f5f7fb; color: #172d47; }
    h1,h2,h3 { color: #142e4f !important; }
    .block-container { max-width: 1300px; padding-top: 2.5rem; }
    [data-testid="stSidebar"] { background: #e9eff6; }
    [data-testid="stForm"] { background: #fff; border-radius: 16px; padding: 1.3rem; }
    .stButton button, .stFormSubmitButton button { border-radius: 9px; }
    .price-card { background: #142e4f; color: white; border-radius: 16px;
                  padding: 1.6rem; margin: 1rem 0; }
    .price-card .price { font-size: clamp(1.8rem, 5vw, 3rem); font-weight: 750; }
    .price-card p { color: #dbe5f0; margin-bottom: 0; }
    [data-testid="stMetric"] { background: white; border: 1px solid #dce4ee;
                               border-radius: 12px; padding: 1rem; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] p { font-size: 1.3rem; white-space: normal;
                                     overflow-wrap: anywhere; text-overflow: clip;
                                     overflow: visible; }
    </style>""", unsafe_allow_html=True)
    st.title("🏠 House Price Prediction Dashboard")
    st.markdown("**AI-Based Regression Dashboard for Estimating Residential Property Valuations**")
    st.write("Enter the property details and click Predict House Price.")

    model_path = resolve_file("house_price_model.pkl")
    model = None
    metadata = {}
    model_key = None
    model_error = None
    if model_path is None:
        model_error = "house_price_model.pkl is missing. Run the training notebook and place its exported model beside app.py."
    else:
        try:
            model_key = fingerprint(model_path)
            model, metadata = load_model(*model_key)
        except Exception as exc:
            model_error = f"The saved model could not be loaded or is incompatible: {exc}"
    if model_error:
        st.error(model_error)
        st.session_state.pop("prediction", None)
    name = estimator_name(model) if model is not None else "Unavailable"

    # Dataset-dependent descriptions are optional; predictions need only the model.
    dataset = None
    dataset_path = resolve_file("01_House_Price.xlsx")
    try:
        if dataset_path is not None:
            dataset = load_dataset(*fingerprint(dataset_path))
            dataset = dataset.rename(columns=metadata.get("column_aliases", {}))
            if dataset.columns.duplicated().any():
                raise ValueError("The dataset contains ambiguous duplicate column names.")
    except Exception as exc:
        dataset = None
        st.warning(f"Dataset could not be loaded: {exc}. Predictions remain available with a valid model.")
    if dataset is None:
        st.info("Dataset guidance and price bands are unavailable. Place 01_House_Price.xlsx "
                "with a Data sheet beside app.py to enable them.")
    thresholds = valuation_thresholds(dataset)
    if dataset is not None and thresholds is None:
        st.info("Price bands are unavailable: house_price_rm is missing or has no valid nonnegative prices.")

    with st.sidebar:
        st.header("About the System")
        st.write("Regression learns the relationship between property characteristics "
                 "and recorded sale prices to estimate a numeric residential valuation.")
        st.subheader("Seven property inputs")
        for label in LABELS.values():
            st.write(f"• {label}")
        st.subheader("Model performance")
        show_performance(metadata, name)
        if metadata.get("data_note"):
            st.info(metadata["data_note"])
        st.caption(DISCLAIMER)

    st.subheader("Property details")
    st.caption("Interface limits: floor area 1–2,000 m²; bedrooms 1–20; bathrooms 1–6; "
               "age 0–50 years; parking spaces 0–4. These are input controls, not training ranges.")
    for column in ("floor_area_m2", "property_age_years"):
        if dataset is not None and column in dataset:
            values = pd.to_numeric(dataset[column], errors="coerce")
            values = values[values.map(lambda x: pd.notna(x) and math.isfinite(float(x)))]
            if not values.empty:
                st.caption(f"Dataset observed {LABELS[column]}: {values.min():g}–{values.max():g}.")
        elif dataset is not None:
            st.caption(f"Dataset range unavailable: missing column {column}.")
        bounds = metadata.get("training_ranges", {}).get(column)
        if bounds:
            st.caption(f"Fitted model's training {LABELS[column]} range: {bounds[0]:g}–{bounds[1]:g}.")

    with st.form("property_form"):
        left, middle, right = st.columns(3)
        with left:
            area = st.number_input("Floor Area (m²)", 1.0, 2000.0, 150.0, 1.0)
            bedrooms = st.number_input("Bedrooms", 1, 20, 3, 1)
            bathrooms = st.number_input("Bathrooms", 1, 6, 2, 1)
        with middle:
            age = st.number_input("Property Age (Years)", 0, 50, 10, 1)
            zone = st.selectbox("Location Zone", ZONES, format_func=lambda x: x.replace("_", " "))
        with right:
            property_type = st.selectbox("Property Type", TYPES, format_func=lambda x: x.replace("_", " "))
            parking = st.number_input("Parking Spaces", 0, 4, 1, 1)
        submitted = st.form_submit_button("Predict House Price", type="primary", disabled=model is None)

    if submitted and model is not None:
        # Store a snapshot: later widget interactions cannot change a previous result.
        st.session_state.pop("prediction", None)
        record = dict(zip(FEATURES, [area, bedrooms, bathrooms, age, zone, property_type, parking]))
        try:
            raw = prepare_inputs([record], model)
            price = predict_prices(model, raw)[0]
            result = {"record": record, "raw": raw, "price": price,
                      "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                      "model_key": model_key, "name": name}
            try:
                result["vector"] = transformed_vector(model, raw)
            except Exception as exc:
                result["vector_error"] = str(exc)
            # Change only location; all other submitted attributes stay fixed.
            try:
                scenarios = prepare_inputs([dict(record, location_zone=z) for z in ZONES], model)
                estimates = predict_prices(model, scenarios)
                baseline = estimates[ZONES.index(zone)]
                result["scenarios"] = pd.DataFrame({
                    "Location Zone": [z.replace("_", " ") for z in ZONES],
                    "Estimated Price (RM)": estimates,
                    "Difference from Baseline (RM)": [value - baseline for value in estimates],
                    "Baseline": ["Selected" if z == zone else "" for z in ZONES],
                })
            except Exception as exc:
                result["scenario_error"] = str(exc)
            st.session_state["prediction"] = result
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")

    result = st.session_state.get("prediction")
    if result and result["model_key"] != model_key:
        st.session_state.pop("prediction", None)
        result = None
        st.info("The model file changed. Submit again to calculate a current estimate.")
    if result:
        record, price = result["record"], result["price"]
        st.markdown(f'<div class="price-card">Estimated house price'
                    f'<div class="price">RM {price:,.2f}</div>'
                    '<p>This is a model estimate in Ringgit Malaysia (RM).</p></div>', unsafe_allow_html=True)
        for column, bounds in metadata.get("training_ranges", {}).items():
            if record[column] < bounds[0] or record[column] > bounds[1]:
                st.warning(f"Submitted {LABELS[column]} is outside the fitted model's training range "
                           f"({bounds[0]:g}–{bounds[1]:g}); the estimate may be less reliable.")
        for column in ("location_zone", "property_type"):
            observed = metadata.get("training_categories", {}).get(column, [])
            if record[column] not in observed:
                st.warning(f"The submitted {LABELS[column]} was not observed in the training partition.")
        a, b, c = st.columns(3)
        a.metric("Price Category", price_category(price, thresholds))
        b.metric("Entered Floor Area (m²)", f"{record['floor_area_m2']:g}")
        c.metric("Model used", result["name"])
        with st.expander("How price categories are defined"):
            if thresholds is None:
                st.write("Unavailable: the dataset's house_price_rm values are required.")
            else:
                q1, q2, q3 = thresholds
                st.write(f"Low: ≤ RM {q1:,.2f}; Moderate: > RM {q1:,.2f} to RM {q2:,.2f}; "
                         f"High: > RM {q2:,.2f} to RM {q3:,.2f}; Very High: > RM {q3:,.2f}.")
                st.caption("Thresholds are the 25th, 50th, and 75th percentiles of valid nonnegative "
                           "house_price_rm values in the supplied Data sheet. These are dataset-relative "
                           "valuation bands, not official bank appraisal tiers.")
        st.subheader("Property summary")
        summary = [{"Item": LABELS[key], "Value": str(record[key]).replace("_", " ")} for key in FEATURES]
        summary += [{"Item": "Estimated Price (RM)", "Value": f"RM {price:,.2f}"},
                    {"Item": "Prediction timestamp (UTC)", "Value": result["timestamp"]}]
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
        st.subheader("Location Zone Comparison (Scenario Analysis)")
        st.caption(f"Selected baseline: {record['location_zone'].replace('_', ' ')}.")
        if "scenarios" in result:
            scenarios = result["scenarios"]
            st.dataframe(scenarios.style.format({"Estimated Price (RM)": "RM {:,.2f}",
                                                "Difference from Baseline (RM)": "RM {:+,.2f}"}),
                         hide_index=True, use_container_width=True)
            st.bar_chart(scenarios.set_index("Location Zone")[["Estimated Price (RM)"]], color="#526f94")
            unobserved = set(ZONES) - set(metadata.get("training_categories", {}).get("location_zone", []))
            if unobserved:
                st.warning("Some scenario zones were absent from the training partition: " + ", ".join(sorted(unobserved)))
        else:
            st.warning("Location comparison unavailable: " + result.get("scenario_error", "Unknown prediction error."))
        st.write("This comparison shows how the trained model’s estimate varies across different "
                 "location zones while keeping physical property attributes constant. It does not "
                 "establish a causal effect or represent a formal appraisal.")
    if model is not None:
        try:
            show_preprocessing(model, metadata, result["raw"] if result else None,
                               result.get("vector") if result else None)
        except Exception as exc:
            st.warning(f"Preprocessing details could not be displayed: {exc}")
        if result and result.get("vector_error"):
            st.caption("Transformed feature display unavailable: " + result["vector_error"])
    st.divider()
    st.caption(DISCLAIMER)
    st.markdown("**Developed by**  \nDanich Khawngam \nNatthanicha Rodtua \nArada Konglek \nTanaphat Chuparat \nPhattharamon Kamonlimsakun \nFaculty of Intelligent Computing \n"
                "Walailak University & Universiti Malaysia Perlis")


if __name__ == "__main__":
    main()
