# supplier_risk_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  # Added for forecast chart
import base64
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression # Added for forecast model
from new_supplier import new_sup

# ------------------------------------
# Page Setup
# ------------------------------------
st.set_page_config(
    page_title="Supplier Risk Management Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------
# Animated Background, 3D Effects & Styling
# ------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(-45deg, #f0f7f4, #dfe7fd, #f7f9fb, #e0f7fa);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    font-family: 'Segoe UI', sans-serif;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
h1 {
    color: #1E3A8A;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.15);
    animation: fadeInDown 1.2s ease;
}
@keyframes fadeInDown {
    0% {opacity: 0; transform: translateY(-30px);}
    100% {opacity: 1; transform: translateY(0);}
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #ffffff, #e6e6e6);
    border-radius: 15px;
    box-shadow: 6px 6px 16px rgba(0,0,0,0.15), -4px -4px 12px rgba(255,255,255,0.8);
    padding: 15px;
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px) scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------
# Add Logo
# ------------------------------------
logo_filename = "lamao.png"
logo_path = os.path.join(os.path.dirname(__file__), logo_filename)

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(
        f"""
        <div style='text-align: center; margin-top: -25px; margin-bottom: 10px;'>
            <img src='data:image/png;base64,{logo_data}' width='420' style='filter: drop-shadow(0 0 10px rgba(0,0,0,0.3));'>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Logo file 'lamao.png' not found — please place it in the same folder as this script.")

# ------------------------------------
# Title and Subtitle
# ------------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>Supplier Risk Management Dashboard</h1>
    <p style='text-align: center; font-size:18px; color: #444;'>
        Gain real-time insights into supplier reliability, quality, and risk performance.
    </p>
    """,
    unsafe_allow_html=True
)

# ------------------------------------
# Sidebar: Data Settings
# ------------------------------------
main_col, right_sidebar = st.columns([4, 1])

with right_sidebar:
    st.header("⚙️ Data Settings")
    num_suppliers = st.number_input("Number of Suppliers", 1, 1000, 5)
    supplier_prefix = st.text_input("Supplier Name Prefix", "Supplier_")
    num_days = st.slider("Number of Days", 7, 90, 30)

    st.markdown("---")
    st.subheader("📊 Score Ranges")

    on_time_min = st.number_input("Min On-Time Delivery (%)", 0.0, 100.0, 80.0)
    on_time_max = st.number_input("Max On-Time Delivery (%)", 0.0, 100.0, 100.0)
    quality_min = st.number_input("Min Quality Score (%)", 0.0, 100.0, 70.0)
    quality_max = st.number_input("Max Quality Score (%)", 0.0, 100.0, 100.0)
    accuracy_min = st.number_input("Min Order Accuracy (%)", 0.0, 100.0, 75.0)
    accuracy_max = st.number_input("Max Order Accuracy (%)", 0.0, 100.0, 100.0)
    compliance_prob = st.slider("Compliance Probability", 0.0, 1.0, 0.95)

# ------------------------------------
# Generate Random Supplier Data
# ------------------------------------
with main_col:
    np.random.seed(42)
    suppliers = [f"{supplier_prefix}{i+1}" for i in range(num_suppliers)]
    dates = pd.date_range(datetime.now() - timedelta(days=num_days), periods=num_days)

    data = [
        [
            supplier,
            date,
            np.random.uniform(on_time_min, on_time_max),
            np.random.uniform(quality_min, quality_max),
            np.random.uniform(accuracy_min, accuracy_max),
            np.random.choice([1, 0], p=[compliance_prob, 1 - compliance_prob])
        ]
        for supplier in suppliers for date in dates
    ]

    df = pd.DataFrame(data, columns=["Supplier", "Date", "On-Time Delivery", "Quality Score", "Order Accuracy", "Compliance"])

    # Filters
    st.header("🔍 Filter Options")
    supplier_filter = st.multiselect("Select Supplier(s)", suppliers, default=suppliers)
    date_range = st.date_input("Select Date Range", [dates.min(), dates.max()])

    filtered_df = df[
        (df['Supplier'].isin(supplier_filter)) &
        (df['Date'] >= pd.Timestamp(date_range[0])) &
        (df['Date'] <= pd.Timestamp(date_range[1]))
    ]

    # ------------------------------------
    # KPIs
    # ------------------------------------
    st.subheader("📈 Performance Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg On-Time Delivery (%)", f"{filtered_df['On-Time Delivery'].mean():.2f}")
    col2.metric("Avg Quality Score (%)", f"{filtered_df['Quality Score'].mean():.2f}")
    col3.metric("Avg Order Accuracy (%)", f"{filtered_df['Order Accuracy'].mean():.2f}")

    # ------------------------------------
    # Supplier Performance Trends
    # ------------------------------------
    st.subheader("📊 Supplier Performance Trends")
    metric_choice = st.selectbox("Select Metric:", ["On-Time Delivery", "Quality Score", "Order Accuracy"])
    fig = px.line(
        filtered_df,
        x="Date",
        y=metric_choice,
        color="Supplier",
        markers=True,
        title=f"{metric_choice} Over Time"
    )
    fig.update_layout(template="plotly_white", transition_duration=800)
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------
    # 🚨 Risk Alerts
    # ------------------------------------
    st.subheader("🚨 Automated Risk Alerts")
    alerts = filtered_df[
        (filtered_df["On-Time Delivery"] < 85) |
        (filtered_df["Quality Score"] < 80) |
        (filtered_df["Compliance"] == 0)
    ]

    if alerts.empty:
        st.success("✅ No active supplier risk alerts!")
    else:
        st.error(f"⚠️ {len(alerts)} potential risk incidents detected!")
        st.dataframe(alerts[["Supplier", "Date", "On-Time Delivery", "Quality Score", "Compliance"]])

    # ------------------------------------
    # 🔍 Quality Analysis (Pie Chart)
    # ------------------------------------
    st.subheader("🔍 Quality Analysis (Simplified View)")
    quality_summary = filtered_df.groupby("Supplier")["Quality Score"].mean().reset_index()
    quality_summary["Quality Score"] = quality_summary["Quality Score"].round(2)

    st.markdown("This pie chart shows each supplier’s contribution to the overall average quality score.")

    fig2 = px.pie(
        quality_summary,
        names="Supplier",
        values="Quality Score",
        hole=0.35, # This creates the 2D doughnut chart
        title="Supplier Quality Score Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig2.update_traces(textinfo="label+percent", pull=[0.05]*len(quality_summary))
    fig2.update_layout(
        template="plotly_white",
        showlegend=True,
        transition_duration=700,
        margin=dict(t=60, b=20, l=10, r=10)
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------
    # 🌐 Dynamic 3D Supplier Performance Map
    # ------------------------------------
    st.subheader("🌐 3D Supplier Performance Map (Dynamic)")

    performance_summary = (
        filtered_df.groupby("Supplier")[["On-Time Delivery", "Quality Score", "Order Accuracy"]]
        .mean()
        .reset_index()
    )

    colx, coly, colz = st.columns(3)
    x_axis = colx.selectbox("X-axis Metric", ["On-Time Delivery", "Quality Score", "Order Accuracy"], index=0)
    y_axis = coly.selectbox("Y-axis Metric", ["On-Time Delivery", "Quality Score", "Order Accuracy"], index=1)
    z_axis = colz.selectbox("Z-axis Metric", ["On-Time Delivery", "Quality Score", "Order Accuracy"], index=2)

    fig_3d = px.scatter_3d(
        performance_summary,
        x=x_axis, y=y_axis, z=z_axis,
        color="Supplier",
        symbol="Supplier",
        opacity=0.85,
        title=f"3D Supplier Performance Map ({x_axis} vs {y_axis} vs {z_axis})"
    )

    fig_3d.update_layout(
        scene=dict(
            xaxis_title=f"{x_axis} (%)",
            yaxis_title=f"{y_axis} (%)",
            zaxis_title=f"{z_axis} (%)",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))
        ),
        template="plotly_white"
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # ------------------------------------
    # 🤖 Predictive Risk Insights (Forecast)
    # ------------------------------------
    st.subheader("🤖 Predictive Risk Insights (Forecast)")

    # Select supplier & metric for forecasting
    forecast_supplier = st.selectbox("Select Supplier for Forecast", suppliers)
    forecast_metric = st.selectbox("Metric to Forecast", ["On-Time Delivery", "Quality Score", "Order Accuracy"])
    forecast_days = st.slider("Forecast Horizon (Days)", 7, 60, 14)

    supplier_data = filtered_df[filtered_df["Supplier"] == forecast_supplier].copy()
    supplier_data = supplier_data.sort_values("Date")

    if len(supplier_data) > 1: # Need at least 2 points to fit a line
        # Numeric encoding for regression
        supplier_data["DayIndex"] = np.arange(len(supplier_data))

        # Fit a simple linear regression
        model = LinearRegression()
        model.fit(supplier_data[["DayIndex"]], supplier_data[forecast_metric])

        # Future prediction
        future_idx = np.arange(len(supplier_data), len(supplier_data) + forecast_days)
        forecast_vals = model.predict(future_idx.reshape(-1, 1))

        # Confidence interval (simple approximation)
        residuals = supplier_data[forecast_metric] - model.predict(supplier_data[["DayIndex"]])
        std_error = residuals.std()
        upper = forecast_vals + 1.96 * std_error
        lower = forecast_vals - 1.96 * std_error

        forecast_dates = pd.date_range(supplier_data["Date"].iloc[-1] + timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecast": forecast_vals,
            "Upper Bound": upper,
            "Lower Bound": lower
        })

        # Combine actual + forecast for visualization
        combined_df = supplier_data[["Date", forecast_metric]].rename(columns={forecast_metric: "Actual"})
        combined_df = pd.concat([combined_df, forecast_df], ignore_index=True)

        # Plot
        fig_forecast = go.Figure()

        fig_forecast.add_trace(go.Scatter(
            x=supplier_data["Date"], y=supplier_data[forecast_metric],
            mode="lines+markers", name="Actual", line=dict(color="#2563EB", width=3),
            hovertemplate="Date: %{x|%b %d, %Y}<br>Value: %{y:.2f}%<extra></extra>"
        ))

        fig_forecast.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Forecast"],
            mode="lines+markers", name="Forecast", line=dict(color="#10B981", dash="dash", width=3),
            hovertemplate="Predicted: %{y:.2f}%<extra></extra>"
        ))

        fig_forecast.add_trace(go.Scatter(
            x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
            y=pd.concat([forecast_df["Upper Bound"], forecast_df["Lower Bound"][::-1]]),
            fill="toself", fillcolor="rgba(16,185,129,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", name="Confidence Interval"
        ))

        fig_forecast.update_layout(
            title=f"{forecast_supplier} – {forecast_metric} Forecast ({forecast_days} days)",
            xaxis_title="Date",
            yaxis_title=f"{forecast_metric} (%)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=60, b=40, l=40, r=40)
        )

        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.info(f"Not enough data for '{forecast_supplier}' to generate a forecast. Select another supplier or adjust filters.")


    # ------------------------------------
    # 🧾 Custom Supplier Risk Evaluation
    # ------------------------------------
    st.markdown("---")
    st.header("🧾 Custom Supplier Risk Evaluation")

    with st.form("custom_supplier_form"):
        st.subheader("Enter Custom Supplier Data")
        supplier_id = st.text_input("Supplier ID", placeholder="e.g., SUP12345")
        order_id = st.text_input("Order ID", placeholder="e.g., ORD98765")
        delay_days = st.number_input("Delay Days", 0, 60, 2)
        delivery_date = st.date_input("Delivery Date", datetime.today())
        reliability_score = st.slider("Supplier Reliability Score (%)", 0.0, 100.0, 85.0)
        parameter_change = st.slider("Parameter Change Magnitude (%)", 0.0, 100.0, 10.0)
        energy_consumption = st.number_input("Energy Consumption (kWh)", 0.0, value=500.0)

        submitted = st.form_submit_button("✨ Evaluate Risk")

    if submitted:
        risk_level, color, show_ballons = new_sup(delay_days, reliability_score, parameter_change)

        st.markdown("### 📊 Custom Supplier Risk Result")
        st.markdown(
            f"""
            <div class='risk-box' style='padding:20px; border-radius:15px; background-color:#fff; border-left: 8px solid {color};'>
                <h3 style='color:{color}; margin-bottom:10px;'>{risk_level}</h3>
                <p><b>Supplier ID:</b> {supplier_id}</p>
                <p><b>Order ID:</b> {order_id}</p>
                <p><b>Delay Days:</b> {delay_days}</p>
                <p><b>Delivery Date:</b> {delivery_date}</p>
                <p><b>Supplier Reliability:</b> {reliability_score:.1f}%</p>
                <p><b>Parameter Change Magnitude:</b> {parameter_change:.1f}%</p>
                <p><b>Energy Consumption:</b> {energy_consumption:.1f} kWh</p>
                <hr>
            </div>
            """,
            unsafe_allow_html=True
        )

        if show_ballons:
            st.balloons()

        st.success("✅ Risk evaluation complete! Use this insight to guide supplier engagement and quality assurance.")

    # ------------------------------------
    # Footer
    # ------------------------------------
    st.markdown("""
    ---
    💡 *Tip:* Integrate AI-driven scoring for deeper supplier insights.
    """)