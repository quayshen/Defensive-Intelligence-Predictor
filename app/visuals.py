"""Visualization functions for Streamlit app"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_blitz_gauge(probability):
    """Plot blitz probability gauge chart"""
    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={'text': "Blitz Probability"},
        delta={'reference': 50, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#FF6B6B"},
            'steps': [
                {'range': [0, 25], 'color': "#51CF66"},
                {'range': [25, 50], 'color': "#FFD93D"},
                {'range': [50, 75], 'color': "#FFA600"},
                {'range': [75, 100], 'color': "#FF6B6B"}
            ],
            'threshold': {
                'line': {'color': "darkred", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    )])
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=12)
    )
    return fig


def plot_coverage_probabilities(coverage_predictions):
    """Plot coverage type probabilities as bar chart"""
    # Prepare data - coverage_predictions should be a dict-like with probabilities
    if isinstance(coverage_predictions, pd.DataFrame):
        # If it's a dataframe, get the row as dict
        coverage_predictions = coverage_predictions.iloc[0].to_dict()
    
    # Extract coverage types and probabilities
    covers = []
    probs = []
    
    for key, val in coverage_predictions.items():
        if key not in ['coverage_type', 'confidence']:
            covers.append(str(key).replace('_', ' ').title())
            probs.append(float(val) if not pd.isna(val) else 0)
    
    # If no detailed probabilities, create a simple chart
    if not covers:
        coverage_type = coverage_predictions.get('coverage_type', 'Unknown')
        confidence = coverage_predictions.get('confidence', 0.5)
        covers = [str(coverage_type)]
        probs = [confidence]
    
    fig = px.bar(
        x=covers,
        y=probs,
        title="Coverage Shell Probabilities",
        labels={'x': 'Coverage Type', 'y': 'Probability'},
        color=probs,
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Coverage Type",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1])
    )
    fig.update_traces(marker_line_width=0)
    return fig


def plot_blitz_by_down_distance(df):
    """Plot blitz rate by down and distance"""
    if df is None or len(df) == 0:
        return None
    
    # Calculate blitz rate by down
    blitz_by_down = df.groupby('down')['blitz'].mean() * 100
    
    fig = px.bar(
        x=blitz_by_down.index,
        y=blitz_by_down.values,
        title="Blitz Rate by Down",
        labels={'x': 'Down', 'y': 'Blitz Rate (%)'},
        color=blitz_by_down.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=350, showlegend=False)
    return fig


def plot_coverage_distribution(coverage_data):
    """Plot coverage distribution as pie chart"""
    if coverage_data is None or len(coverage_data) == 0:
        return None
    
    # Ensure we have data
    if isinstance(coverage_data, dict):
        coverage_types = list(coverage_data.keys())
        counts = list(coverage_data.values())
    else:
        coverage_types = coverage_data.index.tolist()
        counts = coverage_data.values.tolist()
    
    fig = px.pie(
        values=counts,
        names=coverage_types,
        title="Coverage Distribution",
        hole=0.3
    )
    fig.update_layout(height=400)
    return fig


def plot_feature_importance(feature_names, importances, top_n=10):
    """Plot feature importance bar chart"""
    # Sort features by importance
    sorted_idx = np.argsort(importances)[-top_n:]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    fig = px.bar(
        x=sorted_importances,
        y=sorted_features,
        orientation='h',
        title=f"Top {top_n} Feature Importance",
        labels={'x': 'Importance', 'y': 'Feature'},
        color=sorted_importances,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Importance Score",
        yaxis_title="Feature"
    )
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix heatmap"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    return fig


def plot_roc_curve(y_true, y_proba):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure(
        data=[
            go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC={roc_auc:.3f})', line=dict(color='#1f77b4', width=2)),
            go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(color='gray', dash='dash'))
        ]
    )
    
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        hovermode='closest'
    )
    return fig


def plot_prediction_confidence_comparison(predictions_list):
    """Plot comparison of multiple predictions"""
    if not predictions_list:
        return None
    
    pred_names = [p['name'] for p in predictions_list]
    pred_confidence = [p['confidence'] for p in predictions_list]
    pred_type = [p['type'] for p in predictions_list]
    
    fig = px.bar(
        x=pred_names,
        y=pred_confidence,
        color=pred_type,
        title="Prediction Confidence Comparison",
        labels={'x': 'Prediction', 'y': 'Confidence'},
        height=350
    )
    fig.update_layout(showlegend=True)
    return fig


def plot_heatmap_down_distance(df):
    """Plot blitz rate heatmap by down and distance"""
    if df is None or len(df) == 0:
        return None
    
    # Create pivot table
    df_copy = df.copy()
    df_copy['ydstogo_bin'] = pd.cut(df_copy['ydstogo'], bins=[0, 5, 10, 15, 20, 30])
    
    heatmap_data = df_copy.pivot_table(
        values='blitz',
        index='down',
        columns='ydstogo_bin',
        aggfunc='mean'
    ) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns.astype(str),
        y=heatmap_data.index,
        colorscale='Reds'
    ))
    
    fig.update_layout(
        title="Blitz Rate: Down vs Yards to Go",
        xaxis_title="Yards to Go",
        yaxis_title="Down",
        height=400
    )
    return fig

