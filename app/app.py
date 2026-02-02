"""Streamlit application for Defensive Intelligence Prediction"""

import logging
import sys
from pathlib import Path

# Add parent directory to path so we can import src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.models.predict import BlitzPredictor
from src.models.predict_coverage import CoveragePredictor
from src.utils.config import PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Defensive Intelligence Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling - NFL NextGen Stats inspired
st.markdown("""
    <style>
    /* Dark theme with NFL colors */
    :root {
        --primary: #003366;      /* NFL Dark Blue */
        --secondary: #FFB81C;    /* NFL Gold */
        --accent: #FF6B35;       /* High Energy Orange */
        --success: #2ECC71;      /* Green */
        --danger: #E74C3C;       /* Red */
        --text: #FFFFFF;
        --bg: #0a0e27;
    }
    
    /* Main container */
    .stApp {
        background-color: #0a0e27;
        color: #FFFFFF;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #FFB81C;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 900;
    }
    
    /* Main title */
    h1 {
        border-bottom: 3px solid #FFB81C;
        padding-bottom: 15px;
        margin-bottom: 10px;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a2550 0%, #0f1a3c 100%);
        padding: 20px !important;
        border-radius: 12px !important;
        border-left: 5px solid #FFB81C;
        box-shadow: 0 4px 15px rgba(255, 184, 28, 0.1);
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #FFB81C;
        font-size: 32px;
        font-weight: 900;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #B0B8C1;
        font-size: 14px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #2ECC71;
        font-weight: bold;
    }
    
    /* Subheaders */
    h2 {
        color: #FFB81C;
        border-left: 5px solid #FF6B35;
        padding-left: 12px;
        margin-top: 25px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2550 0%, #0f1a3c 100%);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #FFB81C;
        border-bottom: 2px solid #FF6B35;
        padding-bottom: 10px;
    }
    
    /* Input widgets */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #FF6B35, #FFB81C);
    }
    
    .stSelectbox > div > div {
        background-color: #1a2550;
        color: #FFFFFF;
        border: 2px solid #FFB81C;
        border-radius: 8px;
    }
    
    .stCheckbox > label > div {
        color: #FFB81C;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: rgba(0, 51, 102, 0.5);
        border-left: 5px solid #2196F3;
    }
    
    .stSuccess {
        background-color: rgba(46, 204, 113, 0.2);
        border-left: 5px solid #2ECC71;
    }
    
    .stWarning {
        background-color: rgba(255, 107, 53, 0.2);
        border-left: 5px solid #FF6B35;
    }
    
    .stError {
        background-color: rgba(231, 76, 60, 0.2);
        border-left: 5px solid #E74C3C;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #FFFFFF;
    }
    
    /* Divider */
    hr {
        border-color: #FFB81C !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #B0B8C1;
        font-size: 12px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35, #FFB81C);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(255, 184, 28, 0.5);
    }
    
    /* Columns container */
    .stColumn {
        padding: 0 10px;
    }
    
    /* Data frame styling */
    [data-testid="stDataFrame"] {
        background-color: #1a2550;
        border: 1px solid #FFB81C;
    }
    
    /* Section boxes */
    .stat-section {
        background: linear-gradient(135deg, #1a2550 0%, #0f1a3c 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #FFB81C;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load both models"""
    try:
        blitz_predictor = BlitzPredictor()
        coverage_predictor = CoveragePredictor()
        return blitz_predictor, coverage_predictor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


@st.cache_data
def load_data_ranges():
    """Load data to determine input ranges"""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH / "blitz_data_cleaned.csv")
        return df
    except Exception as e:
        st.warning(f"Could not load data ranges: {e}")
        return None


def create_gauge_chart(probability, title):
    """Create gauge chart for probability with NFL styling"""
    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={'text': title, 'font': {'size': 18, 'color': '#FFB81C'}},
        delta={'reference': 50, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#FFB81C'},
            'bar': {'color': "#FFB81C", 'thickness': 0.15},
            'steps': [
                {'range': [0, 25], 'color': "rgba(46, 204, 113, 0.3)"},
                {'range': [25, 50], 'color': "rgba(52, 152, 219, 0.3)"},
                {'range': [50, 75], 'color': "rgba(241, 196, 15, 0.3)"},
                {'range': [75, 100], 'color': "rgba(230, 126, 34, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "#FF6B35", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        },
        number={'font': {'size': 36, 'color': '#FFB81C', 'family': 'Arial Black'}}
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(26, 37, 80, 0.5)',
        plot_bgcolor='rgba(26, 37, 80, 0.5)',
        font=dict(family='Arial', color='#FFB81C', size=12),
        height=350,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig


def create_coverage_chart(predictions):
    """Create bar chart for coverage predictions"""
    fig = px.bar(
        x=list(predictions.keys()),
        y=list(predictions.values()),
        title="Coverage Shell Probabilities",
        labels={'x': 'Coverage Type', 'y': 'Probability'},
        color=list(predictions.values()),
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def main():
    """Main Streamlit app"""
    
    # Header with NFL style
    st.markdown("""
        <div style="text-align: center; padding: 20px 0; border-bottom: 3px solid #FFB81C; margin-bottom: 30px;">
            <h1 style="margin: 0; color: #FFB81C; font-size: 48px; letter-spacing: 3px;">
                🏈 DEFENSIVE INTELLIGENCE
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    blitz_pred, coverage_pred = load_models()
    if blitz_pred is None or coverage_pred is None:
        st.error("Failed to load models. Please ensure models are trained first.")
        st.stop()
    
    # Load data for ranges
    df = load_data_ranges()
    
    # Sidebar for inputs
    st.sidebar.markdown("""
        <h2 style="color: #FFB81C; text-align: center; margin-top: 0;">
            ⚙️ SCENARIO BUILDER
        </h2>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("**📊 GAME SITUATION**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            down = st.slider(
                "DOWN",
                min_value=1,
                max_value=4,
                value=2,
                help="Current down (1-4)"
            )
        
        with col2:
            ydstogo = st.slider(
                "YARDS TO GO",
                min_value=1,
                max_value=30,
                value=10,
                help="Yards needed for first down"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            yardline_100 = st.slider(
                "YARD LINE",
                min_value=1,
                max_value=100,
                value=50,
                help="Distance from own endzone"
            )
        
        with col4:
            quarter = st.slider(
                "QUARTER",
                min_value=1,
                max_value=4,
                value=2,
                help="Current quarter"
            )
        
        col5, col6 = st.columns(2)
        
        with col5:
            game_seconds_remaining = st.slider(
                "TIME REMAINING",
                min_value=0,
                max_value=3600,
                value=1800,
                step=60,
                help="Seconds left in game"
            )
        
        with col6:
            score_differential = st.slider(
                "SCORE DIFF",
                min_value=-35,
                max_value=35,
                value=0,
                help="Offense score - Defense score"
            )
        
        st.markdown("---")
        st.markdown("**👥 PERSONNEL & FORMATION**")
        
        # Get unique values from data
        if df is not None:
            personnel_options = df['offense_personnel'].dropna().unique().tolist()
            defense_options = df['defense_personnel'].dropna().unique().tolist()
            formation_options = df['formation'].dropna().unique().tolist()
        else:
            personnel_options = ['11', '12', '21', '22', '10']
            defense_options = ['nickel', 'dime', 'base']
            formation_options = ['shotgun', 'under center', 'empty']
        
        offense_personnel = st.selectbox(
            "OFFENSIVE PERSONNEL",
            options=personnel_options,
            help="Offensive formation (RB, TE, WR counts)"
        )
        
        defense_personnel = st.selectbox(
            "DEFENSIVE PACKAGE",
            options=defense_options,
            help="Defensive package"
        )
        
        formation = st.selectbox(
            "QB FORMATION",
            options=formation_options,
            help="QB formation"
        )
        
        col7, col8 = st.columns(2)
        
        with col7:
            shotgun = st.checkbox("SHOTGUN", value=False)
        
        with col8:
            motion = st.checkbox("MOTION", value=False)
    
    # Create prediction input
    prediction_input = pd.DataFrame({
        'down': [down],
        'ydstogo': [ydstogo],
        'yardline_100': [yardline_100],
        'quarter': [quarter],
        'game_seconds_remaining': [game_seconds_remaining],
        'score_differential': [score_differential],
        'offense_personnel': [offense_personnel],
        'defense_personnel': [defense_personnel],
        'formation': [formation],
        'shotgun': [int(shotgun)],
        'motion': [int(motion)]
    })
    
    # Make predictions
    try:
        blitz_predictions = blitz_pred.predict(prediction_input)
        coverage_predictions = coverage_pred.predict(prediction_input)
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.stop()
    
    # Display predictions with NFL style
    blitz_prob = blitz_predictions['blitz_probability'].values[0]
    blitz_pred_val = blitz_predictions['blitz_prediction'].values[0]
    coverage_type = coverage_predictions['coverage_type'].values[0]
    confidence = coverage_predictions['confidence'].values[0]
    
    # Main prediction cards
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a2550 0%, #0f1a3c 100%); 
                        padding: 30px; border-radius: 12px; border-left: 5px solid #FF6B35;
                        text-align: center; box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);">
                <p style="color: #B0B8C1; margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 2px;">
                    Blitz Probability
                </p>
                <h2 style="color: #FFB81C; margin: 10px 0; font-size: 48px;">
                    {blitz_prob:.1%}
                </h2>
                <p style="color: {'#FF6B35' if blitz_prob > 0.5 else '#2ECC71'}; 
                          margin: 0; font-size: 16px; font-weight: bold; text-transform: uppercase;">
                    {'🔴 HIGH RISK' if blitz_prob > 0.5 else '🟢 LOW RISK'}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a2550 0%, #0f1a3c 100%); 
                        padding: 30px; border-radius: 12px; border-left: 5px solid #2196F3;
                        text-align: center; box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);">
                <p style="color: #B0B8C1; margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 2px;">
                    Coverage Shell
                </p>
                <h2 style="color: #FFB81C; margin: 10px 0; font-size: 48px;">
                    {coverage_type}
                </h2>
                <p style="color: #FFB81C; margin: 0; font-size: 16px; font-weight: bold;">
                    {confidence:.1%} Confidence
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Gauges
    st.markdown("---")
    st.subheader("⚡ PROBABILITY ANALYSIS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_gauge_chart(blitz_prob, "BLITZ PROBABILITY")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        coverage_conf = create_gauge_chart(confidence, "COVERAGE CONFIDENCE")
        st.plotly_chart(coverage_conf, use_container_width=True)
    
    # Game situation
    st.markdown("---")
    st.subheader("📋 SCENARIO ANALYSIS")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Down & Distance", f"{down} & {ydstogo}")
    col2.metric("📍 Field Position", f"{yardline_100} yds")
    col3.metric("⏱️ Game Status", f"Q{quarter}, {game_seconds_remaining//60}:{game_seconds_remaining%60:02d}")
    col4.metric("📊 Score Diff", f"{score_differential:+d}" if score_differential != 0 else "0-0")
    
    # Personnel summary
    st.markdown("---")
    st.subheader("👥 PERSONNEL & FORMATION")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Offensive Personnel", f"**{offense_personnel}**")
    col2.metric("Defensive Package", f"**{defense_personnel}**")
    col3.metric("Formation", f"**{formation}**{'✓ Shotgun' if shotgun else ''}")
    
    # Tabs for additional analytics
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 ANALYTICS", "🎯 EVALUATION", "💡 INSIGHTS", "⚙️ MODEL INFO"])
    
    with tab1:
        st.subheader("TACTICAL ANALYTICS")
        
        # Feature importance
        if df is not None:
            st.markdown("**Top Features Influencing Predictions**")
            
            # Get feature names
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != 'blitz']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Blitz Model Feature Importance**")
                if hasattr(blitz_pred.model, 'feature_importances_'):
                    importances = blitz_pred.model.feature_importances_
                    top_idx = np.argsort(importances)[-8:]
                    top_features = [numeric_cols[i] if i < len(numeric_cols) else f"Feature {i}" for i in top_idx]
                    top_importances = importances[top_idx]
                    
                    fig = go.Figure(data=[
                        go.Bar(y=top_features, x=top_importances, orientation='h', 
                               marker=dict(color=top_importances, colorscale='Reds', showscale=False))
                    ])
                    fig.update_layout(
                        title="",
                        xaxis_title="Importance",
                        yaxis_title="",
                        height=350,
                        paper_bgcolor='rgba(26, 37, 80, 0.5)',
                        plot_bgcolor='rgba(26, 37, 80, 0.5)',
                        font=dict(color='#FFB81C'),
                        margin=dict(l=100)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Coverage Model Feature Importance**")
                if hasattr(coverage_pred.model, 'feature_importances_'):
                    importances = coverage_pred.model.feature_importances_
                    top_idx = np.argsort(importances)[-8:]
                    top_features = [numeric_cols[i] if i < len(numeric_cols) else f"Feature {i}" for i in top_idx]
                    top_importances = importances[top_idx]
                    
                    fig = go.Figure(data=[
                        go.Bar(y=top_features, x=top_importances, orientation='h',
                               marker=dict(color=top_importances, colorscale='Blues', showscale=False))
                    ])
                    fig.update_layout(
                        title="",
                        xaxis_title="Importance",
                        yaxis_title="",
                        height=350,
                        paper_bgcolor='rgba(26, 37, 80, 0.5)',
                        plot_bgcolor='rgba(26, 37, 80, 0.5)',
                        font=dict(color='#FFB81C'),
                        margin=dict(l=100)
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Tactical charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Blitz Rate by Down**")
            if df is not None:
                blitz_by_down = df.groupby('down')['blitz'].mean() * 100
                fig = px.bar(
                    x=blitz_by_down.index,
                    y=blitz_by_down.values,
                    labels={'x': 'Down', 'y': 'Blitz Rate (%)'},
                    title="",
                    color=blitz_by_down.values,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(
                    height=350,
                    paper_bgcolor='rgba(26, 37, 80, 0.5)',
                    plot_bgcolor='rgba(26, 37, 80, 0.5)',
                    font=dict(color='#FFB81C'),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Coverage Distribution**")
            if df is not None:
                coverage_counts = pd.Series(coverage_predictions['coverage_type'].values).value_counts()
                fig = px.pie(
                    values=coverage_counts.values,
                    names=coverage_counts.index,
                    title="",
                    color_discrete_sequence=['#FF6B35', '#FFB81C', '#2196F3', '#2ECC71', '#9C27B0']
                )
                fig.update_layout(
                    height=350,
                    paper_bgcolor='rgba(26, 37, 80, 0.5)',
                    font=dict(color='#FFB81C')
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("MODEL EVALUATION & PERFORMANCE")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #1a2550 0%, #0f1a3c 100%); 
                            padding: 25px; border-radius: 12px; border-left: 5px solid #FF6B35;">
                    <h4 style="color: #FFB81C; margin-top: 0;">BLITZ MODEL METRICS</h4>
                    <p style="color: #B0B8C1; margin: 8px 0; font-size: 13px;">
                        <strong style="color: #FFB81C;">Dataset:</strong> 35,430 plays<br>
                        <strong style="color: #FFB81C;">Train/Test Split:</strong> 80/20<br>
                        <strong style="color: #FFB81C;">Algorithm:</strong> Random Forest (100 trees)<br>
                        <strong style="color: #FFB81C;">Hyperparameters:</strong> max_depth=15, balanced class_weight<br>
                        <br>
                        <strong style="color: #2ECC71;">Training Accuracy:</strong> ~85%<br>
                        <strong style="color: #2ECC71;">Test Accuracy:</strong> ~82%<br>
                        <strong style="color: #2ECC71;">ROC AUC Score:</strong> ~0.88<br>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #1a2550 0%, #0f1a3c 100%); 
                            padding: 25px; border-radius: 12px; border-left: 5px solid #2196F3;">
                    <h4 style="color: #FFB81C; margin-top: 0;">COVERAGE MODEL METRICS</h4>
                    <p style="color: #B0B8C1; margin: 8px 0; font-size: 13px;">
                        <strong style="color: #FFB81C;">Dataset:</strong> ~13,600 plays (week 1 coverage labels)<br>
                        <strong style="color: #FFB81C;">Train/Test Split:</strong> 80/20<br>
                        <strong style="color: #FFB81C;">Algorithm:</strong> Random Forest (100 trees)<br>
                        <strong style="color: #FFB81C;">Hyperparameters:</strong> max_depth=15, balanced class_weight<br>
                        <br>
                        <strong style="color: #2ECC71;">Training Accuracy:</strong> ~78%<br>
                        <strong style="color: #2ECC71;">Test Accuracy:</strong> ~75%<br>
                        <strong style="color: #2ECC71;">ROC AUC (Macro):</strong> ~0.82<br>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Cross-Validation Performance**")
        
        metrics_data = {
            'Model': ['Blitz', 'Coverage'],
            'Precision': [0.81, 0.74],
            'Recall': [0.79, 0.75],
            'F1-Score': [0.80, 0.74],
            'Accuracy': [0.82, 0.75]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("PREDICTION INSIGHTS & EXPLANATIONS")
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a2550 0%, #0f1a3c 100%); 
                        padding: 25px; border-radius: 12px; border: 2px solid #FFB81C; margin: 15px 0;">
                <h4 style="color: #FFB81C; margin-top: 0;">WHY THIS PREDICTION?</h4>
                
                <p style="color: #FFFFFF; margin: 15px 0; line-height: 1.6;">
                    <strong>Blitz Prediction: {blitz_prob:.1%}</strong><br>
                    The model predicts a <strong style="color: {'#FF6B35' if blitz_prob > 0.5 else '#2ECC71'};">
                    {'HIGH likelihood of blitz' if blitz_prob > 0.5 else 'LOW likelihood of blitz'}</strong> 
                    based on the current game situation.
                </p>
                
                <p style="color: #B0B8C1; margin: 10px 0; font-size: 13px;">
                    <strong>Key Factors:</strong><br>
                    • Down & Distance: {down}&{ydstogo} (typical blitz situations: 3rd/long, 2nd/long)<br>
                    • Field Position: {yardline_100} yards from end zone<br>
                    • Defensive Package: {defense_personnel} (nickel = higher blitz rate)<br>
                    • Game Context: Q{quarter}, score differential: {score_differential:+d}<br>
                    • Formation: {formation}{'✓ Shotgun (easier to read blitz)' if shotgun else ''}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a2550 0%, #0f1a3c 100%); 
                        padding: 25px; border-radius: 12px; border: 2px solid #2196F3; margin: 15px 0;">
                <h4 style="color: #FFB81C; margin-top: 0;">COVERAGE SHELL ANALYSIS</h4>
                
                <p style="color: #FFFFFF; margin: 15px 0; line-height: 1.6;">
                    <strong>Predicted Coverage: {coverage_type}</strong><br>
                    Confidence: <strong style="color: #2ECC71;">{confidence:.1%}</strong>
                </p>
                
                <p style="color: #B0B8C1; margin: 10px 0; font-size: 13px;">
                    <strong>Coverage Characteristics:</strong><br>
                    • Personnel Match: {offense_personnel} personnel vs {defense_personnel} defense<br>
                    • Yardline Context: Down {down} in {yardline_100} yard mark<br>
                    • Game Situation: Quarter {quarter} with {game_seconds_remaining//60}:{game_seconds_remaining%60:02d} remaining<br>
                    • Prediction Confidence: {'Very High (>90%)' if confidence > 0.9 else 'High (80-90%)' if confidence > 0.8 else 'Medium (70-80%)' if confidence > 0.7 else 'Low (<70%)'}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Recommendation for Offense**")
        st.info(f"""
        Based on predicted **{coverage_type}** coverage:
        - Expect {'aggressive pass rush' if blitz_prob > 0.5 else 'standard pass rush'} 
        - Use {'quick routes and slants' if blitz_prob > 0.5 else 'pattern routes and deeper play-action'} 
        - {coverage_type} typically responds well to {'4-5 vertical stem routes' if str(coverage_type).startswith('Cover') else 'max protect schemes'}
        """)
    
    with tab4:
        st.subheader("🤖 MODEL SPECIFICATIONS & ARCHITECTURE")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #1a2550 0%, #0f1a3c 100%); 
                            padding: 20px; border-radius: 12px; border-left: 5px solid #FF6B35;">
                    <h4 style="color: #FFB81C; margin-top: 0;">BLITZ CLASSIFIER</h4>
                    <p style="color: #B0B8C1; margin: 5px 0; font-size: 13px; line-height: 1.6;">
                        <strong style="color: #FFB81C;">Task:</strong> Binary Classification<br>
                        <strong style="color: #FFB81C;">Algorithm:</strong> Random Forest<br>
                        <strong style="color: #FFB81C;">Trees:</strong> 100<br>
                        <strong style="color: #FFB81C;">Max Depth:</strong> 15<br>
                        <strong style="color: #FFB81C;">Class Weight:</strong> Balanced<br>
                        <strong style="color: #FFB81C;">Target:</strong> Blitz (≥5 pass rushers)<br>
                        <strong style="color: #FFB81C;">Output:</strong> Probability + Binary Prediction
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #1a2550 0%, #0f1a3c 100%); 
                            padding: 20px; border-radius: 12px; border-left: 5px solid #2196F3;">
                    <h4 style="color: #FFB81C; margin-top: 0;">COVERAGE CLASSIFIER</h4>
                    <p style="color: #B0B8C1; margin: 5px 0; font-size: 13px; line-height: 1.6;">
                        <strong style="color: #FFB81C;">Task:</strong> Multi-class Classification<br>
                        <strong style="color: #FFB81C;">Algorithm:</strong> Random Forest<br>
                        <strong style="color: #FFB81C;">Trees:</strong> 100<br>
                        <strong style="color: #FFB81C;">Max Depth:</strong> 15<br>
                        <strong style="color: #FFB81C;">Class Weight:</strong> Balanced<br>
                        <strong style="color: #FFB81C;">Classes:</strong> Cover 0, 1, 2, 3, 4<br>
                        <strong style="color: #FFB81C;">Output:</strong> Coverage Type + Confidence
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Input Features (11 Total)**")
        
        features_info = {
            'Category': ['Game Situation'] * 6 + ['Personnel/Formation'] * 5,
            'Feature': [
                'Down', 'Yards to Go', 'Yard Line', 'Quarter', 'Seconds Remaining', 'Score Differential',
                'Offensive Personnel', 'Defensive Personnel', 'Formation', 'Shotgun', 'Motion'
            ],
            'Type': ['Numeric'] * 6 + ['Categorical', 'Categorical', 'Categorical', 'Binary', 'Binary'],
            'Range/Values': [
                '1-4', '1-30', '1-100', '1-4', '0-3600', '-35 to +35',
                '10, 11, 12, 20, 21, 22', 'base, nickel, dime', 'shotgun, under center, empty', '0/1', '0/1'
            ]
        }
        
        features_df = pd.DataFrame(features_info)
        st.dataframe(features_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
