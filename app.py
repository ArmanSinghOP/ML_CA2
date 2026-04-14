import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ------------------ CONFIG ------------------
st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

# ------------------ STYLE ------------------
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] > div {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 12px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("🚀 Interactive ML Pipeline Dashboard")

# ------------------ FILE UPLOAD ------------------
st.sidebar.header("1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if 'Product_id' in df.columns:
        df = df.drop('Product_id', axis=1)
    return df

if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.success("File Uploaded!")
else:
    st.warning("Upload dataset to proceed")
    st.stop()

# ------------------ SAFE NUMERIC DATA ------------------
numeric_df = df.select_dtypes(include='number')

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data & EDA",
    "Cleaning",
    "Feature Selection",
    "Model Training",
    "Performance"
])

# ================== TAB 1 ==================
with tab1:
    st.subheader("📊 Exploratory Data Analysis")

    # -------- DATA SUMMARY --------
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 📋 Dataset Summary")
        st.dataframe(df.describe())

    with col2:
        st.write("### 🔥 Correlation Heatmap")
        fig = px.imshow(numeric_df.corr())
        st.plotly_chart(fig, use_container_width=True)

    # -------- TARGET SELECTION --------
    target_eda = st.selectbox(
        "Select Target Variable",
        numeric_df.columns,
        key="eda_target"
    )

    # -------- FEATURE SELECTION --------
    feature = st.selectbox(
        "Select Feature",
        numeric_df.columns,
        key="eda_feature"
    )

    # -------- FEATURE DISTRIBUTION --------
    st.write(f"### 📊 Distribution of {feature}")
    fig = px.histogram(df, x=feature)
    st.plotly_chart(fig, use_container_width=True)

    # -------- TARGET DISTRIBUTION --------
    st.write(f"### 🎯 Distribution of Target ({target_eda})")
    fig_target = px.histogram(df, x=target_eda)
    st.plotly_chart(fig_target, use_container_width=True)

    # -------- FEATURE VS TARGET --------
    st.write(f"### 🔗 {feature} vs {target_eda}")
    fig_scatter = px.scatter(
        df,
        x=feature,
        y=target_eda,
        title=f"{feature} vs {target_eda}"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ================== TAB 2 ==================
with tab2:
    st.subheader("🧹 Cleaning, Scaling & Outliers")

    df = st.session_state.get("df", df)

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    # -------- SCALING --------
    scaler = MinMaxScaler()

    scale_cols = st.multiselect(
        "Columns to Scale",
        numeric_df.columns,
        default=[col for col in ['Sale','battery'] if col in numeric_df.columns],
        key="scale_cols"
    )

    if st.button("Apply Scaling"):
        df_scaled = df.copy()
        for col in scale_cols:
            df_scaled[col] = scaler.fit_transform(df[[col]])
        st.session_state["df"] = df_scaled
        st.success("Scaling Applied!")

    # -------- OUTLIER DETECTION --------
    st.write("### 🚨 Outlier Detection (IQR Method)")

    outlier_col = st.selectbox(
        "Select Column for Outlier Detection",
        numeric_df.columns,
        key="outlier_col"
    )

    Q1 = df[outlier_col].quantile(0.25)
    Q3 = df[outlier_col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[outlier_col] < lower) | (df[outlier_col] > upper)]

    st.write(f"Detected Outliers: {len(outliers)}")

    fig = px.box(df, y=outlier_col, title="Box Plot (Outliers)")
    st.plotly_chart(fig)

    if st.button("Remove Outliers"):
        df_clean = df[(df[outlier_col] >= lower) & (df[outlier_col] <= upper)]
        st.session_state["df"] = df_clean
        st.success(f"Outliers removed! New shape: {df_clean.shape}")

# ================== TAB 3 ==================
with tab3:
    st.subheader("🎯 Feature Selection")

    df = st.session_state.get("df", df)

    target_fs = st.selectbox(
        "Select Target Variable",
        numeric_df.columns,
        key="fs_target"
    )

    st.session_state["target"] = target_fs

    X = df.drop(target_fs, axis=1)
    y = df[target_fs]

    # Correlation
    st.write("### 📊 Correlation with Target")
    corr = df.corr()[target_fs].sort_values(ascending=False)

    fig_corr = px.bar(corr, title="Correlation with Target")
    st.plotly_chart(fig_corr)

    # Feature Importance
    st.write("### 🌳 Feature Importance")
    model = RandomForestRegressor()
    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_imp = px.bar(importance, x="Feature", y="Importance")
    st.plotly_chart(fig_imp)

    st.dataframe(importance)

    use_top_features = st.checkbox("Use Top Important Features Only")

    if use_top_features:
        selected_features = list(importance["Feature"].head(5))
        st.info("Using top 5 important features automatically")
    else:
        selected_features = st.multiselect(
            "Select Features",
            X.columns,
            default=list(X.columns),
            key="feature_select"
        )

    if len(selected_features) == 0:
        st.warning("Select at least one feature!")
        st.stop()

    st.session_state["features"] = selected_features

# ================== TAB 4 ==================
with tab4:
    st.subheader("🤖 Model Training")

    df = st.session_state.get("df", df)

    target = st.session_state.get("target", numeric_df.columns[-1])
    features = st.session_state.get("features", numeric_df.columns[:-1])

    X = df[features]
    y = df[target]

    test_size = st.slider("Test Size", 0.1, 0.5, 0.3, key="test_size")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model_name = st.selectbox(
        "Select Model",
        ["Linear Regression", "KNN", "Decision Tree", "Random Forest"],
        key="model_select"
    )

    if model_name == "Linear Regression":
        model = LinearRegression()

    elif model_name == "KNN":
        k = st.slider("K Value", 1, 20, 11, key="knn_k")
        model = KNeighborsRegressor(n_neighbors=k)

    elif model_name == "Decision Tree":
        depth = st.slider("Depth", 1, 20, 9, key="dt_depth")
        model = DecisionTreeRegressor(max_depth=depth)

    else:
        model = RandomForestRegressor()

    if st.button("🚀 Train Model"):
        try:
            model.fit(X_train, y_train)

            st.session_state["model"] = model
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test

            st.success("Model Trained Successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

# ================== TAB 5 ==================
with tab5:
    st.subheader("📈 Performance")

    if "model" in st.session_state:
        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        col1, col2, col3 = st.columns(3)

        col1.metric("📉 MAE", f"{mae:.2f}")
        col2.metric("📊 MSE", f"{mse:.2f}")
        col3.metric("📈 R²", f"{r2:.3f}")

        st.write("### Actual vs Predicted")

        fig = px.scatter(
            x=y_test,
            y=pred,
            labels={'x': "Actual", 'y': "Predicted"},
            title="Actual vs Predicted"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Train a model first!")  