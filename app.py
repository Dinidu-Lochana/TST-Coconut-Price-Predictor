import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import altair as alt
from model import TimeSeriesTransformer, TransformerBlock, PositionalEncoding

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Coconut Price Predictor",
    page_icon="🥥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .hero-section {
        padding: 0rem 0.5rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        text-align: center;
        background: none;
    }
    
    .hero-title {
        font-size: 5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }
    
    .custom-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    .upload-section {
        background: linear-gradient(135deg, rgba(103, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .prediction-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        border-color: #10b981;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(16, 185, 129, 0.2);
    }
    
    .prediction-week {
        font-size: 1.2rem;
        color: #718096;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .prediction-price {
        font-size: 1.4rem;
        font-weight: 700;
        color: #10b981;
        margin-bottom: 0.3rem;
    }
    
    .prediction-confidence {
        font-size: 1.2rem;
        font-weight: 400;
        color: #047857;
        margin-bottom: 0.3rem;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        text-align: center;
        margin: 1.5rem 0 1.5rem 0;
        position: relative;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: 0px;
        left: 50%;
        transform: translateX(-50%);
        width: 50px;
        height: 3px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    
    .alert-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .alert-error {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .download-section {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 2.5rem 0;
        border: 20px solid #f59e0b;
    }
    
    .download-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #92400e;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model and Scalers
# -----------------------------
@st.cache_resource
def load_model_and_scalers():
    model = tf.keras.models.load_model(
        "model/coconut_transformer.keras",
        custom_objects={
            "TimeSeriesTransformer": TimeSeriesTransformer,
            "TransformerBlock": TransformerBlock,
            "PositionalEncoding": PositionalEncoding
        }
    )
    with open("model/feature_scaler.joblib", "rb") as f:
        feature_scaler = joblib.load(f)
    with open("model/target_scaler.joblib", "rb") as f:
        target_scaler = joblib.load(f)
    return model, feature_scaler, target_scaler

model, feature_scaler, target_scaler = load_model_and_scalers()

# -----------------------------
# Feature columns
# -----------------------------
feature_cols = [
    "Crop Report Weekly","Weekly Coconut Price","Export Crop Weekly",
    "Dessicated coconut (Indonesia)","Dessicated coconut (Phillipines)",
    "Temperature (Celcius)","Rainfall (mm)","Public Holidays",
    "National Level elections (Presidential or General election)",
    "Actual Crop Production previous 24th week","Actual Crop Production previous 20th week",
    "Actual Crop Production previous 16th week","Actual Crop Production previous 12th week",
    "Actual Crop Production previous 8th week","Actual Crop Production previous 4th week",
    "Predicted Crop Production next 4th week","Predicted Crop Production next 8th week",
    "Predicted Crop Production next 12th week","Predicted Crop Production next 16th week",
    "Predicted Crop Production next 20th week","Predicted Crop Production next 24th week"
]

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">🥥 Coconut Price Predictor</h1>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.image("KAP.png", width=180)

    st.markdown("## 📁 File Upload")
    st.markdown("Upload your Coconut Price Dataset")
    
    uploaded_file = st.file_uploader("Choose your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded!")
        
        tab_selection = st.radio(
            label="",
            options=["Dashboard", "Dataset Overview"],
            index=0,
            label_visibility="collapsed"
        )    

# -----------------------------
# Main Content
# -----------------------------
if uploaded_file is not None:
    # Ensure required features exist
    missing = [col for col in feature_cols if col not in df.columns]
    
    if missing:
        st.markdown(f'<div class="alert-error">❌ Missing required columns: {", ".join(missing[:5])}</div>', unsafe_allow_html=True)
    else:
        if tab_selection == "Dashboard":
            # Processing
            with st.spinner('🤖 Processing data...'):
                # Fetch last 20 rows
                first_20 = df[feature_cols].iloc[-21:-1].values
                last_20 = df[feature_cols].tail(20).values
                
                # Scale features
                scaled_first_20 = feature_scaler.transform(first_20)
                scaled_first_20 = np.expand_dims(scaled_first_20, axis=0)

                scaled_last_20 = feature_scaler.transform(last_20)
                scaled_last_20 = np.expand_dims(scaled_last_20, axis=0)
                
                # Predict
                pred_first_scaled = model.predict(scaled_first_20)
                pred_first = target_scaler.inverse_transform(pred_first_scaled)

                pred_scaled = model.predict(scaled_last_20)
                pred = target_scaler.inverse_transform(pred_scaled)

                # Actual price of current week
                actual_last_week_price = df["Weekly Coconut Price"].iloc[-1]

                # Predicted price for the current week
                predicted_last_week_price = pred_first[0][0]

                # Residual correction
                residual_correction = actual_last_week_price - predicted_last_week_price

                pred = pred + residual_correction

                # Confidence intervals
                conf_75_intervals = [
                    5.575526713, 8.336572388, 12.10769231, 15.05043059,
                    16.76105385, 19.47060216, 20.78810314, 23.22679757,
                    25.79191728, 26.74781699, 28.91333343, 31.64232587
                ]

                conf_95_intervals = [
                    9.516969015, 14.22984862, 20.66684253, 25.68985657,
                    28.60975085, 33.23472866, 35.4835953, 39.64624764,
                    44.02469762, 45.65634041, 49.35270021, 54.01086757
                ]

            # Chart Section
            if "Date" not in df.columns:
                st.markdown('<div class="alert-error">❌ Date column required for visualization</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="section-title">📊 Price Trend Analysis</div>', unsafe_allow_html=True)
                
                df["Date"] = pd.to_datetime(df["Date"])

                # Prepare chart data
                last_20_dates = df["Date"].tail(20).values
                last_20_prices = df["Weekly Coconut Price"].tail(20).values
                last_date = df["Date"].iloc[-1]
                last_price = df["Weekly Coconut Price"].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=12, freq="W")

                # Create dataframes
                actual_df = pd.DataFrame({
                    "Date": last_20_dates,
                    "Price": last_20_prices,
                    "Type": "Actual Data"
                })

                predicted_df = pd.DataFrame({
                    "Date": future_dates,
                    "Price": pred[0],
                    "Type": "Prediction Data"
                })

                predicted_df["Price_lower"] = predicted_df["Price"] - conf_75_intervals
                predicted_df["Price_upper"] = predicted_df["Price"] + conf_75_intervals

                predicted_df["Price_lower_95"] = predicted_df["Price"] - conf_95_intervals
                predicted_df["Price_upper_95"] = predicted_df["Price"] + conf_95_intervals

                # Connector data
                connector_df = pd.DataFrame({
                    "Date": [last_date, future_dates[0]],
                    "Price": [last_price, pred[0][0]],
                    "Type": ["Actual Data", "Prediction Data"]
                })

                connector_ci_df = pd.DataFrame({
                    "Date": [last_date, future_dates[0]],
                    "Price_lower": [last_price, pred[0][0] - conf_75_intervals[0]],
                    "Price_upper": [last_price, pred[0][0] + conf_75_intervals[0]]
                })

                connector_ci_95_df = pd.DataFrame({
                    "Date": [last_date, future_dates[0]],
                    "Price_lower_95": [last_price, pred[0][0] - conf_95_intervals[0]],
                    "Price_upper_95": [last_price, pred[0][0] + conf_95_intervals[0]]
                })
                
                # Main line chart
                chart_main = (
                    alt.Chart(pd.concat([actual_df, predicted_df]))
                    .mark_line(point=True, strokeWidth=3)
                    .encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y("Price:Q", title="Coconut Nut Price (LKR)"),
                        color=alt.Color(
                            "Type:N",
                            scale=alt.Scale(domain=["Actual Data", "Prediction Data"], range=["#3b82f6", "#10b981"]),
                            legend=alt.Legend(title="Data Type", orient="bottom")
                        ),
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date"),
                            alt.Tooltip("Price:Q", title="Price (LKR)", format=".2f"),
                            alt.Tooltip("Type:N", title="Data Type")
                        ]
                    )
                )

                # Confidence interval
                chart_ci = (
                    alt.Chart(predicted_df)
                    .mark_area(opacity=0.2, color="#10b981")
                    .encode(
                        x="Date:T",
                        y="Price_lower:Q",
                        y2="Price_upper:Q",
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date"),
                            alt.Tooltip("Price_lower:Q", title="Lower Bound (LKR)", format=".2f"),
                            alt.Tooltip("Price_upper:Q", title="Upper Bound (LKR)", format=".2f"),                      
                        ]
                    )
                )

                # Confidence interval 95%
                chart_ci_95 = (
                    alt.Chart(predicted_df)
                    .mark_area(opacity=0.1, color="#10b981")
                    .encode(
                        x="Date:T",
                        y="Price_lower_95:Q",
                        y2="Price_upper_95:Q",
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date"),
                            alt.Tooltip("Price_lower_95:Q", title="95% Lower Bound (LKR)", format=".2f"),
                            alt.Tooltip("Price_upper_95:Q", title="95% Upper Bound (LKR)", format=".2f"),                      
                        ]
                    )
                )

                # Connector line
                chart_connector_line = (
                    alt.Chart(connector_df)
                    .mark_line(color="#10b981", strokeWidth=3)
                    .encode(x="Date:T", y="Price:Q")
                )

                # Connector CI
                chart_connector_ci = (
                    alt.Chart(connector_ci_df)
                    .mark_area(opacity=0.25, color="#10b981")
                    .encode(x="Date:T", y="Price_lower:Q", y2="Price_upper:Q")
                )

                # Connector CI 95%
                chart_connector_ci_95 = (
                    alt.Chart(connector_ci_95_df)
                    .mark_area(opacity=0.15, color="#10b981")
                    .encode(x="Date:T", y="Price_lower_95:Q", y2="Price_upper_95:Q")
                )

                # Combine all layers
                final_chart = (
                    chart_connector_ci_95 + chart_ci_95 + chart_connector_ci + chart_ci + chart_connector_line + chart_main
                ).properties(
                    width=700,
                    height=400,
                    title="Coconut Price Forecast Analysis"
                ).resolve_scale(
                    color='independent'
                )
                
                st.altair_chart(final_chart, use_container_width=True)

                # Weekly Predictions
                st.markdown('<div class="section-title">🔮 12-Week Coconut Price Forecast</div>', unsafe_allow_html=True)
                
                cols = st.columns(3)
                for i, (price, ci) in enumerate(zip(pred[0], conf_75_intervals), start=1):
                    col_idx = (i - 1) % 3
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-week">WEEK {i}</div>
                            <div style="display: flex; justify-content: center; gap: 0.5rem; align-items: baseline;">
                                <div class="prediction-confidence">LKR per Nut : </div>
                                <div class="prediction-price">{price:.2f}</div>
                                <div class="prediction-confidence">± {ci:.2f}</div>
                            </div>
                            <div style="display: flex; justify-content: center; gap: 0.5rem; align-items: baseline;">
                                <div class="prediction-confidence">LKR per Kg&nbsp;&nbsp;&nbsp;: </div>
                                <div class="prediction-price">{(price/0.65):.2f}</div>
                                <div class="prediction-confidence">± {(ci/0.65):.2f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br><br>", unsafe_allow_html=True)

                # Download Section
                predicted_csv_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Coconut Price (LKR per Nut)": pred[0],
                    "Lower Bound (LKR per Nut)": pred[0] - conf_75_intervals,
                    "Upper Bound (LKR per Nut)": pred[0] + conf_75_intervals,
                    "Predicted Coconut Price (LKR per Kg)": pred[0]/0.65,
                    "Lower Bound (LKR per Kg)": (pred[0] - conf_75_intervals)/0.65,
                    "Upper Bound (LKR per Kg)": (pred[0] + conf_75_intervals)/0.65
                })

                csv_bytes = predicted_csv_df.to_csv(index=False).encode('utf-8')
             
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label="📥 Download Prediction Report",
                        data=csv_bytes,
                        file_name=f"coconut_forecast_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                

        elif tab_selection == "Dataset Overview":
            # Dataset Overview Tab
            st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
            
            with st.container():
                st.dataframe(df.tail(21), use_container_width=True, height=300)
                
else:
    # Welcome message when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #6b7280;">
        <h3>Welcome to Coconut Price Predictor!</h3>
    </div>
    """, unsafe_allow_html=True)