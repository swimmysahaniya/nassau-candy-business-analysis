import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.markdown(
    """
    <h2 style='text-align:center;'>
    🍬 Nassau Candy Business Intelligence Dashboard
    </h2>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    page_title="Advanced Profitability Dashboard",
    layout="wide"
)


# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "nassau_candy_distributor.csv")

    df = pd.read_csv(csv_path)

    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

    df['Month'] = df['Order Date'].dt.month
    df['Year'] = df['Order Date'].dt.year

    df['Margin'] = df['Gross Profit'] / df['Sales']

    return df


df = load_data()

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("🔎 Filters")

region = st.sidebar.multiselect(
    "Select Region",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

division = st.sidebar.multiselect(
    "Select Division",
    options=df['Division'].unique(),
    default=df['Division'].unique()
)

year = st.sidebar.multiselect(
    "Select Year",
    options=df['Year'].unique(),
    default=df['Year'].unique()
)

filtered_df = df[
    (df['Region'].isin(region)) &
    (df['Division'].isin(division)) &
    (df['Year'].isin(year))
].copy()

selected_product = st.sidebar.selectbox(
    "Search Product",
    filtered_df['Product Name'].unique()
)

product_data = filtered_df[
    filtered_df['Product Name'] == selected_product
]

total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Gross Profit'].sum()
avg_margin = filtered_df['Margin'].mean()

st.sidebar.metric(
    "💰 Total Sales",
    f"${total_sales:,.0f}"
)

st.sidebar.metric(
    "📈 Total Profit",
    f"${total_profit:,.0f}"
)

st.sidebar.metric(
    "📊 Avg Margin",
    f"{avg_margin:.2f}"
)

# -----------------------------------
# EXECUTIVE SUMMARY
# -----------------------------------
st.title("📊 Advanced Product Profitability Dashboard")

st.info("""
### 📌 Executive Summary

- Wonka Bar products dominate profitability
- Chocolate division is the strongest business segment
- Pacific region generates highest profit
- Several products contribute very low profit
- Sales strongly influence profitability trends
""")

# -----------------------------------
# KPI SECTION
# -----------------------------------
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Gross Profit'].sum()
avg_margin = filtered_df['Margin'].mean()

if avg_margin < 0.3:
    st.warning("⚠️ Profit margins are low. Consider reducing operational costs.")
else:
    st.success("✅ Profit margins are healthy.")

previous_profit = df[df['Year'] == 2024]['Gross Profit'].sum()

growth = (
    ((total_profit - previous_profit) / previous_profit) * 100
    if previous_profit != 0 else 0
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "💰 Total Sales",
    f"${total_sales:,.0f}"
)

col2.metric(
    "📈 Total Profit",
    f"${total_profit:,.0f}",
    f"{growth:.2f}%"
)

col3.metric(
    "📊 Avg Margin",
    f"{avg_margin:.2f}"
)

# -----------------------------------
# TOP PRODUCT CARD
# -----------------------------------
best_product = (
    filtered_df.groupby('Product Name')['Gross Profit']
    .sum()
    .sort_values(ascending=False)
    .idxmax()
)

best_profit = (
    filtered_df.groupby('Product Name')['Gross Profit']
    .sum()
    .max()
)

st.success(
    f"🏆 Best Performing Product: {best_product} | Profit: ${best_profit:,.0f}"
)

# -----------------------------------
# TABS
# -----------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Analytics",
    "🤖 Machine Learning",
    "📋 Insights"
])

# ===================================
# TAB 1 - ANALYTICS
# ===================================
with tab1:

    # -----------------------------------
    # PROFIT CATEGORY
    # -----------------------------------
    st.subheader("📊 Profit Distribution")

    filtered_df['Profit Category'] = pd.qcut(
        filtered_df['Gross Profit'],
        q=4,
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    profit_counts = (
        filtered_df['Profit Category']
        .value_counts()
        .reset_index()
    )

    profit_counts.columns = ['Category', 'Count']

    fig = px.bar(
        profit_counts,
        x='Category',
        y='Count',
        title='Profit Category Distribution',
        template='plotly_dark',
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------
    # TOP PRODUCTS
    # -----------------------------------
    st.subheader("🏆 Top 10 Products")

    product_profit = (
        filtered_df.groupby('Product Name')['Gross Profit']
        .sum()
    )

    top_products = (
        product_profit
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_products,
        x='Gross Profit',
        y='Product Name',
        orientation='h',
        template='plotly_dark',
        title='Top 10 Products by Profit'
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------
    # BOTTOM PRODUCTS
    # -----------------------------------
    st.subheader("⚠️ Bottom 10 Products")

    bottom_products = (
        filtered_df.groupby('Product Name')['Margin']
        .mean()
        .sort_values()
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        bottom_products,
        x='Margin',
        y='Product Name',
        orientation='h',
        template='plotly_dark',
        title='Bottom 10 Products by Profit'
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------
    # REGION PERFORMANCE
    # -----------------------------------
    st.subheader("🌍 Region Performance")

    region_profit = (
        filtered_df.groupby('Region')['Gross Profit']
        .sum()
        .reset_index()
    )

    fig = px.bar(
        region_profit,
        x='Region',
        y='Gross Profit',
        template='plotly_dark',
        title='Region-wise Profit'
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------
    # DIVISION SHARE
    # -----------------------------------
    st.subheader("🥧 Division Profit Share")

    division_profit = (
        filtered_df.groupby('Division')['Gross Profit']
        .sum()
        .reset_index()
    )

    fig = px.pie(
        division_profit,
        names='Division',
        values='Gross Profit',
        template='plotly_dark',
        title='Division-wise Profit Contribution',
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------
    # CORRELATION MATRIX
    # -----------------------------------
    st.subheader("📌 Correlation Matrix")

    corr = filtered_df[
        ['Sales', 'Cost', 'Gross Profit', 'Units']
    ].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        template='plotly_dark',
        title="Feature Correlation"
    )

    st.plotly_chart(fig, use_container_width=True)

# ===================================
# TAB 2 - MACHINE LEARNING
# ===================================
with tab2:

    st.subheader("🤖 Profit Prediction Model")

    filtered_df['Profit Class'] = pd.qcut(
        filtered_df['Gross Profit'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    X = filtered_df[['Sales', 'Cost', 'Units', 'Month', 'Year']]
    y = filtered_df['Profit Class']

    # -----------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------------
    # MODEL TRAINING
    # -----------------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # -----------------------------------
    # PREDICTIONS
    # -----------------------------------
    y_pred = model.predict(X_test)

    # -----------------------------------
    # MODEL ACCURACY
    # -----------------------------------
    accuracy = accuracy_score(y_test, y_pred)

    st.metric(
        "✅ Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    health_score = (
            avg_margin * 40 +
            accuracy * 40 +
            growth / 5
    )

    st.sidebar.metric("🏥 Business Health Score", f"{health_score:.1f}/100")

    # -----------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------
    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='Blues',
        template='plotly_dark',
        labels=dict(x="Predicted", y="Actual"),
        x=['Low', 'Medium', 'High'],
        y=['Low', 'Medium', 'High']
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------
    st.write("### 🔍 Feature Importance")

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values()

    importance_df = importance.reset_index()
    importance_df.columns = ['Feature', 'Importance']

    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        template='plotly_dark',
        title='Feature Importance'
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------
    # PREDICTION TOOL
    # -----------------------------------
    st.subheader("🎯 Predict Profit")

    sales_input = st.number_input(
        "Enter Sales ($)",
        min_value=0.0,
        value=1000.0
    )

    cost_input = st.number_input(
        "Enter Cost ($)",
        min_value=0.0,
        value=500.0
    )

    units_input = st.number_input(
        "Enter Units Sold",
        min_value=1.0,
        value=10.0
    )

    month_input = st.slider(
        "Month",
        1,
        12,
        6
    )

    year_input = st.slider(
        "Year",
        int(df['Year'].min()),
        int(df['Year'].max()),
        int(df['Year'].max())
    )

    if st.button("Predict Profit"):

        input_data = pd.DataFrame({
            'Sales': [sales_input],
            'Cost': [cost_input],
            'Units': [units_input],
            'Month': [month_input],
            'Year': [year_input]
        })

        input_data = input_data[X.columns]

        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        confidence = probabilities.max() * 100

        st.info(
            f"Model Confidence: {confidence:.2f}%"
        )

        st.success(
            f"Predicted Profit Category: {prediction[0]}"
        )

        prob_df = pd.DataFrame({
            'Category': model.classes_,
            'Probability': probabilities[0]
        })

        fig = px.bar(
            prob_df,
            x='Category',
            y='Probability',
            color='Category',
            template='plotly_dark',
            title='Prediction Confidence'
        )

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------
    # FORECAST + ACTUAL TREND
    # -----------------------------------
    st.subheader("📈 Profit Trend Forecast")

    # Monthly Actual Profit
    monthly_forecast = (
        filtered_df.groupby('Month')['Gross Profit']
        .sum()
        .reset_index()
    )

    # Train Forecast Model
    X_forecast = monthly_forecast[['Month']]
    y_forecast = monthly_forecast['Gross Profit']

    forecast_model = LinearRegression()
    forecast_model.fit(X_forecast, y_forecast)

    # Future Months
    future_months = pd.DataFrame({
        'Month': [13, 14, 15]
    })

    future_predictions = forecast_model.predict(future_months)

    # Actual Data
    actual_df = pd.DataFrame({
        'Month': monthly_forecast['Month'],
        'Profit': monthly_forecast['Gross Profit'],
        'Type': 'Actual'
    })

    # Forecast Data
    forecast_df = pd.DataFrame({
        'Month': [13, 14, 15],
        'Profit': future_predictions,
        'Type': 'Forecast'
    })

    # Combine Data
    combined_df = pd.concat([actual_df, forecast_df])

    # Create Graph
    fig = px.line(
        combined_df,
        x='Month',
        y='Profit',
        color='Type',
        markers=True,
        template='plotly_dark',
        title='Actual vs Forecasted Profit Trend'
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Profit",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# ===================================
# TAB 3 - INSIGHTS
# ===================================
with tab3:

    st.subheader("🧠 Business Insights")

    st.write("""
    ### Key Findings

    - Wonka Bar products contribute the highest overall profit
    - Chocolate division is the strongest-performing category
    - Pacific region generates the highest regional profit
    - Several products fall into low-profit segments
    - Sales strongly influence profitability trends
    - Profit increases significantly during later months
    """)

    st.subheader("💼 Recommendations")

    st.write("""
    - Increase investment in high-performing chocolate products
    - Optimize or discontinue low-profit products
    - Expand operations in Pacific region
    - Improve margin through cost optimization
    - Focus on seasonal demand forecasting
    """)

# -----------------------------------
# DOWNLOAD BUTTON
# -----------------------------------
st.download_button(
    label="📥 Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_profitability_data.csv",
    mime="text/csv"
)

# -----------------------------------
# DATA PREVIEW
# -----------------------------------
st.subheader("📋 Dataset Preview")

st.dataframe(filtered_df.head(50))

# -----------------------------------
# PREMIUM FOOTER
# -----------------------------------
st.markdown("""
<style>
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #0E1117;
color: gray;
text-align: center;
padding: 10px;
font-size: 14px;
border-top: 1px solid #333;
z-index: 100;
}
</style>

<div class="footer">
📊 Nassau Candy Business Intelligence Dashboard |
Built with ❤️ using Streamlit & Plotly |
🚀 Developed by <b>Swimmy Sahaniya</b>
</div>
""", unsafe_allow_html=True)