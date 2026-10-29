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



# Original supplied CIVIL logo embedded so app.py needs no extra branding assets.
LOGO_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAMACAIAAAA12IJaAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAFVGlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSfvu78nIGlkPSdXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQnPz4KPHg6eG1wbWV0YSB4bWxuczp4PSdhZG9iZTpuczptZXRhLyc+CjxyZGY6UkRGIHhtbG5zOnJkZj0naHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyc+CgogPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9JycKICB4bWxuczpBdHRyaWI9J2h0dHA6Ly9ucy5hdHRyaWJ1dGlvbi5jb20vYWRzLzEuMC8nPgogIDxBdHRyaWI6QWRzPgogICA8cmRmOlNlcT4KICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0nUmVzb3VyY2UnPgogICAgIDxBdHRyaWI6Q3JlYXRlZD4yMDI2LTEwLTI5PC9BdHRyaWI6Q3JlYXRlZD4KICAgICA8QXR0cmliOkRhdGE+eyZxdW90O2RvYyZxdW90OzomcXVvdDtEQUcxNU1Ta05TMCZxdW90OywmcXVvdDt1c2VyJnF1b3Q7OiZxdW90O1VBRW5fbWlVb1ljJnF1b3Q7LCZxdW90O2JyYW5kJnF1b3Q7OiZxdW90O0JBRW5fdUwtcGEwJnF1b3Q7fTwvQXR0cmliOkRhdGE+CiAgICAgPEF0dHJpYjpFeHRJZD40YjUxNzllMS00NzlkLTRiZTUtYmYwYS02N2MxYTM1YWNjZDA8L0F0dHJpYjpFeHRJZD4KICAgICA8QXR0cmliOkZiSWQ+NTI1MjY1OTE0MTc5NTgwPC9BdHRyaWI6RmJJZD4KICAgICA8QXR0cmliOlRvdWNoVHlwZT4yPC9BdHRyaWI6VG91Y2hUeXBlPgogICAgPC9yZGY6bGk+CiAgIDwvcmRmOlNlcT4KICA8L0F0dHJpYjpBZHM+CiA8L3JkZjpEZXNjcmlwdGlvbj4KCiA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0nJwogIHhtbG5zOmRjPSdodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4x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IH/rQS6LBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYCAAD//+3XgQAAAACAIH/rQS6LBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgQAAABGBAAAAEYEAAAARgJ2N/YVFhtdGAAAAABJRU5ErkJggg=="
# Precompiled Tailwind utilities; no CDN or Node installation needed to run this app.
BRAND_UTILITIES = '/*! tailwindcss v4.3.3 | MIT License | https://tailwindcss.com */\n:root,:host{--spacing:.25rem;--radius-2xl:1rem}.flex{display:flex}.items-center{align-items:center}.gap-4{gap:calc(var(--spacing) * 4)}.rounded-2xl{border-radius:var(--radius-2xl)}.p-6{padding:calc(var(--spacing) * 6)}'

# Lucide house icon, distributed under the ISC license:
# ISC License
# 
# Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
# 
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
# 
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
HOUSE_ICON = '<svg aria-hidden="true" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/><path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>'


def apply_branding():
    st.markdown('<style>' + BRAND_UTILITIES + """
    :root { --civil-navy:#011939; --civil-gold:#dfb522; }
    .stApp { background:#f8f7f3; color:#132a42; }
    [data-testid="stHeader"] { background:#f8f7f3; }
    .block-container { max-width:1300px; padding-top:2rem; padding-bottom:3rem; }
    h1,h2,h3 { color:#011939 !important; font-family:Georgia,serif !important;
               font-weight:400 !important; letter-spacing:-.03em; }
    [data-testid="stCaptionContainer"] { color:#627181; }
    .civil-header { background:#011939; color:#f8f7f3; margin-bottom:2rem;
                    border-bottom:3px solid #dfb522; }
    .civil-logo { width:74px; height:74px; overflow:hidden; position:relative; flex-shrink:0; }
    .civil-logo img { width:205px; max-width:none; position:absolute; left:50%; top:50%;
                      transform:translate(-50%,-46%); }
    .civil-header h1 { color:#f7f1de !important; margin:0; padding:0;
                       font-size:2.25rem !important; letter-spacing:.13em; }
    .civil-header p { color:#d4bd72; font-size:.72rem; letter-spacing:.18em;
                     text-transform:uppercase; margin:.4rem 0 0; }
    .civil-eyebrow { color:#7d682a; font-size:.72rem; letter-spacing:.2em; margin-bottom:.6rem; }
    .civil-section { display:flex; align-items:center; gap:12px; color:#9b7b20; margin:1.6rem 0 .5rem; }
    .civil-section h2 { margin:0; padding:0; font-size:1.65rem; }
    [data-testid="stSidebar"] { background:#011939; border-right:1px solid #253a52; }
    [data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:#e8cc7a !important; }
    [data-testid="stSidebar"] p,[data-testid="stSidebar"] li,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#d1dbe6; }
    [data-testid="stSidebar"] [data-testid="stAlert"] { background:#183049; border:1px solid #34506a; }
    [data-testid="stSidebar"] button { color:#e9cd7a; }
    [data-testid="stForm"] { background:#fff; border:1px solid #e6e3d8;
        border-radius:16px; padding:1.5rem; box-shadow:0 5px 22px #01193906; }
    [data-testid="stForm"] label { color:#253c51; }
    [data-testid="stForm"] input { color:#011939; background:#fafaf7; }
    [data-baseweb="select"] > div { background:#fafaf7; color:#011939; }
    .stButton button,.stFormSubmitButton button { border-radius:8px; }
    .stFormSubmitButton button[kind="primary"] { background:#011939; color:#f7e6a6;
        border:1px solid #011939; min-height:46px; padding:.65rem 1.5rem; }
    .stFormSubmitButton button[kind="primary"]:hover { background:#17344e; border-color:#dfb522; }
    button:focus-visible,input:focus-visible { outline:2px solid #b18b18 !important; outline-offset:3px; }
    .price-card { background:#011939; color:#d2deec; border:1px solid #203b54;
        border-top:3px solid #dfb522; border-radius:16px; padding:2rem; margin:1.5rem 0; }
    .price-card .price { color:#efd071; font-family:Georgia,serif;
        font-size:clamp(2rem,5vw,3.5rem); font-weight:400; margin:.5rem 0; }
    .price-card p { color:#c2cedd; margin-bottom:0; }
    [data-testid="stMetric"] { background:#fff; border:1px solid #e6e3d8;
        border-radius:12px; padding:1.1rem; }
    [data-testid="stMetricValue"],[data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] p { font-size:1.3rem; white-space:normal;
        overflow-wrap:anywhere; text-overflow:clip; overflow:visible; color:#011939; }
    [data-testid="stExpander"] { background:#fffefa; border-color:#e6e3d8; }
    @media(max-width:640px) {
      .block-container { padding:1.25rem 1rem; }
      .civil-header { padding:1.1rem; gap:12px; }
      .civil-header h1 { font-size:1.8rem !important; }
      .civil-header p { font-size:.6rem; letter-spacing:.1em; }
      .civil-logo { width:58px; height:62px; }
      [data-testid="stForm"] { padding:1rem; }
    }
    </style>""", unsafe_allow_html=True)
    st.markdown(
        '<div class="civil-header flex items-center gap-4 rounded-2xl p-6">'
        f'<div class="civil-logo"><img src="{LOGO_DATA_URI}" alt="CIVIL gold gear logo"></div>'
        '<div><h1>CIVIL</h1><p>Property Intelligence</p></div></div>'
        '<div class="civil-eyebrow">RESIDENTIAL INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="CIVIL | House Price Prediction", page_icon="🏠", layout="wide")
    apply_branding()
    st.title("Property valuation")
    st.write("A considered estimate. Enter your property details to explore its predicted value.")

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
        st.header("CIVIL workspace")
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

    st.markdown(f'<div class="civil-section">{HOUSE_ICON}<h2>Property details</h2></div>',
                unsafe_allow_html=True)
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
            st.bar_chart(scenarios.set_index("Location Zone")[["Estimated Price (RM)"]], color="#b79535")
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
    st.markdown("**Developed by**  \nDanich Khawngam  \nNatthanicha Rodtua  \nArada Konglek  \nTanaphat Chuparat  \nPhattharamon Kamonlimsakun  \nWalailak University & Universiti Malaysia Perlis")


if __name__ == "__main__":
    main()
