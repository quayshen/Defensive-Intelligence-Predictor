"""UI components for Streamlit app"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def create_header():
    """Create header component"""
    st.markdown("# 🏈 Defensive Intelligence Predictor")
    st.markdown("**Advanced defensive coverage prediction powered by ML**")
    st.markdown("---")


def create_sidebar_inputs():
    """Create sidebar input controls and return as dictionary"""
    st.sidebar.title("📋 Game Situation")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        down = st.slider(
            "Down",
            min_value=1,
            max_value=4,
            value=2,
            help="Current down (1-4)"
        )
    
    with col2:
        ydstogo = st.slider(
            "Yards to Go",
            min_value=1,
            max_value=30,
            value=10,
            help="Yards needed for first down"
        )
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        yardline_100 = st.slider(
            "Yards from Endzone",
            min_value=1,
            max_value=100,
            value=50,
            help="Distance from own endzone"
        )
    
    with col4:
        quarter = st.slider(
            "Quarter",
            min_value=1,
            max_value=4,
            value=2,
            help="Current quarter"
        )
    
    col5, col6 = st.sidebar.columns(2)
    with col5:
        game_seconds_remaining = st.slider(
            "Seconds Remaining",
            min_value=0,
            max_value=3600,
            value=1800,
            step=60,
            help="Seconds left in game"
        )
    
    with col6:
        score_differential = st.slider(
            "Score Differential",
            min_value=-35,
            max_value=35,
            value=0,
            help="Offense score - Defense score"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.title("👥 Personnel & Formation")
    
    personnel_options = ['11', '12', '21', '22', '10']
    defense_options = ['nickel', 'dime', 'base']
    formation_options = ['shotgun', 'under center', 'empty']
    
    offense_personnel = st.sidebar.selectbox(
        "Offensive Personnel",
        options=personnel_options,
        help="Offensive formation (RB, TE, WR counts)"
    )
    
    defense_personnel = st.sidebar.selectbox(
        "Defensive Personnel",
        options=defense_options,
        help="Defensive package"
    )
    
    formation = st.sidebar.selectbox(
        "Formation",
        options=formation_options,
        help="QB formation"
    )
    
    col7, col8 = st.sidebar.columns(2)
    with col7:
        shotgun = st.checkbox("Shotgun", value=False)
    
    with col8:
        motion = st.checkbox("Motion", value=False)
    
    return {
        'down': down,
        'ydstogo': ydstogo,
        'yardline_100': yardline_100,
        'quarter': quarter,
        'game_seconds_remaining': game_seconds_remaining,
        'score_differential': score_differential,
        'offense_personnel': offense_personnel,
        'defense_personnel': defense_personnel,
        'formation': formation,
        'shotgun': int(shotgun),
        'motion': int(motion)
    }


def display_blitz_prediction(blitz_prob, blitz_pred):
    """Display blitz prediction with metrics"""
    st.subheader("⚡ Blitz Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Blitz Probability",
            f"{blitz_prob:.1%}",
            delta=f"{'HIGH RISK' if blitz_prob > 0.5 else 'LOW RISK'}",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Prediction",
            "🔴 BLITZ" if blitz_pred else "🟢 NO BLITZ",
        )
    
    with col3:
        st.metric(
            "Confidence",
            f"{max(blitz_prob, 1 - blitz_prob):.1%}"
        )


def display_coverage_prediction(coverage_type, confidence):
    """Display coverage prediction with metrics"""
    st.subheader("🛡️ Coverage Shell Prediction")
    
    # Map cover types to descriptions
    cover_descriptions = {
        'Cover 0': 'Man coverage with no deep safety help',
        'Cover 1': 'Man coverage with one deep safety',
        'Cover 2': 'Two-deep safeties with zone underneath',
        'Cover 3': 'Three-deep with zone underneath',
        'Cover 4': 'Four-deep (quarters) coverage'
    }
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.metric(
            "Predicted Coverage",
            coverage_type,
            delta=f"{confidence:.1%} confidence"
        )
        st.caption(cover_descriptions.get(str(coverage_type), ""))
    
    with col2:
        # Color-coded confidence indicator
        if confidence > 0.8:
            st.success(f"High Confidence: {confidence:.1%}")
        elif confidence > 0.6:
            st.info(f"Medium Confidence: {confidence:.1%}")
        else:
            st.warning(f"Low Confidence: {confidence:.1%}")


def display_game_situation(situation_dict):
    """Display game situation summary"""
    st.markdown("---")
    st.subheader("📊 Game Situation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Down & Distance", f"{situation_dict['down']} & {situation_dict['ydstogo']}")
    col2.metric("Field Position", f"{situation_dict['yardline_100']} yds")
    col3.metric("Game Status", f"Q{situation_dict['quarter']}, {situation_dict['game_seconds_remaining']//60} min")
    
    score_diff = situation_dict['score_differential']
    col4.metric("Score Diff", f"+{score_diff}" if score_diff >= 0 else f"{score_diff}")
    
    # Personnel display
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Offensive Personnel", situation_dict['offense_personnel'])
    col2.metric("Defensive Package", situation_dict['defense_personnel'])
    
    formation_info = ""
    if situation_dict['shotgun']:
        formation_info += "Shotgun "
    if situation_dict['motion']:
        formation_info += "+ Motion"
    col3.metric("Formation", f"{situation_dict['formation']} {formation_info}")


def display_model_info():
    """Display model information"""
    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Blitz Model:**
        - Algorithm: Random Forest (100 trees)
        - Task: Binary Classification
        - Predicts: Blitz (5+ pass rushers) vs No Blitz
        - Features: 11 game situation + personnel features
        """)
    
    with col2:
        st.info("""
        **Coverage Model:**
        - Algorithm: Random Forest (100 trees)
        - Task: Multi-class Classification
        - Predicts: Cover 0, 1, 2, 3, or 4
        - Features: Same as Blitz Model
        """)

