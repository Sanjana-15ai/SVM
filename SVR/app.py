import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="SVR Predictor", page_icon="🤖", layout="wide")

st.title("🤖 SVR Regression — Train & Predict")
st.markdown("Upload your dataset, train a model, then enter values manually to get a prediction.")

# ── Session state ──────────────────────────────────────────────────────────────
for key in ["model", "scaler", "features", "target", "trained", "metrics"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "trained" not in st.session_state:
    st.session_state.trained = False

# ── Sidebar — Upload & Config ──────────────────────────────────────────────────
st.sidebar.header("1️⃣ Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()

raw_data = pd.read_csv(uploaded_file)
numeric_data = raw_data.select_dtypes(include=[np.number])

st.sidebar.header("2️⃣ Model Settings")
all_cols = numeric_data.columns.tolist()
target = st.sidebar.selectbox("Target column (y)", all_cols, index=len(all_cols) - 1)
features = st.sidebar.multiselect("Feature columns (X)", [c for c in all_cols if c != target],
                                  default=[c for c in all_cols if c != target])
model_choice = st.sidebar.selectbox("Model", ["SVR", "Linear Regression"])
if model_choice == "SVR":
    kernel  = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    C       = st.sidebar.number_input("C", 0.01, 1000.0, 1.0)
    epsilon = st.sidebar.number_input("Epsilon", 0.001, 10.0, 0.1)
test_size  = st.sidebar.slider("Test split", 0.1, 0.4, 0.2, 0.05)
scale_data = st.sidebar.checkbox("Scale features (recommended for SVR)", value=True)
remove_out = st.sidebar.checkbox("Remove outliers (IQR)", value=True)

# ── Outlier removal ────────────────────────────────────────────────────────────
if remove_out:
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)
    clean_data = numeric_data[mask]
else:
    clean_data = numeric_data.copy()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "🏋️ Train Model", "🎯 Predict"])

# ── Tab 1 · Data Preview ───────────────────────────────────────────────────────
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(raw_data))
    col2.metric("After Outlier Removal", len(clean_data))
    col3.metric("Features Selected", len(features))

    st.dataframe(clean_data.head(20), use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    corr = clean_data.corr()
    cax = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax)
    for (i, j), val in np.ndenumerate(corr.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color="white" if abs(val) > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    st.pyplot(fig)

# ── Tab 2 · Train ──────────────────────────────────────────────────────────────
with tab2:
    if not features:
        st.warning("Select at least one feature in the sidebar.")
        st.stop()

    if st.button("🚀 Train Model", type="primary"):
        X = clean_data[features].values
        y = clean_data[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        scaler = None
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        if model_choice == "SVR":
            model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.model    = model
        st.session_state.scaler   = scaler
        st.session_state.features = features
        st.session_state.target   = target
        st.session_state.trained  = True
        st.session_state.metrics  = {
            "mse":    mean_squared_error(y_test, y_pred),
            "rmse":   np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2":     r2_score(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
        }
        st.success(f"✅ {model_choice} trained successfully!")

    if st.session_state.trained:
        m = st.session_state.metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE",      f"{m['mse']:.4f}")
        c2.metric("RMSE",     f"{m['rmse']:.4f}")
        c3.metric("R² Score", f"{m['r2']:.4f}")

        st.subheader("Actual vs Predicted")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.scatter(m["y_test"], m["y_pred"], alpha=0.6, edgecolors="k", linewidths=0.4)
        lims = [min(m["y_test"].min(), m["y_pred"].min()),
                max(m["y_test"].max(), m["y_pred"].max())]
        ax2.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        ax2.legend()
        st.pyplot(fig2)

# ── Tab 3 · Predict ────────────────────────────────────────────────────────────
with tab3:
    if not st.session_state.trained:
        st.warning("⚠️ Please train the model first in the **Train Model** tab.")
    else:
        st.subheader("Enter Feature Values")
        st.markdown(f"**Target to predict:** `{st.session_state.target}`")

        input_vals = {}
        num_feats = len(st.session_state.features)
        cols = st.columns(min(num_feats, 3))
        for i, feat in enumerate(st.session_state.features):
            col_min  = float(clean_data[feat].min())
            col_max  = float(clean_data[feat].max())
            col_mean = float(clean_data[feat].mean())
            with cols[i % 3]:
                input_vals[feat] = st.number_input(
                    f"{feat}",
                    value=round(col_mean, 4),
                    help=f"Range in training data: {col_min:.2f} – {col_max:.2f}"
                )

        if st.button("🎯 Predict", type="primary"):
            input_array = np.array([[input_vals[f] for f in st.session_state.features]])

            if st.session_state.scaler:
                input_array = st.session_state.scaler.transform(input_array)

            prediction = st.session_state.model.predict(input_array)[0]

            st.success(f"### Predicted **{st.session_state.target}** = `{prediction:.4f}`")

            st.markdown("**Your Input Summary:**")
            summary_df = pd.DataFrame([input_vals])
            st.dataframe(summary_df, use_container_width=True)