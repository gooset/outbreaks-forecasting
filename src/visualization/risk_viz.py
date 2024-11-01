import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import folium
from folium.plugins import MarkerCluster
import plotly.io as pio

def create_interactive_risk_choropleth(df):
    """
    Create an interactive choropleth map of arbovirus risk scores
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing city-level risk assessment data
    
    Returns:
    --------
    go.Figure
        Interactive Plotly choropleth map
    """
    # Prepare data for choropleth
    fig = px.scatter_geo(
        df, 
        lon='longitude', 
        lat='latitude',
        color='risk_score',
        size='risk_score',
        hover_name='city',
        hover_data={
            'risk_score': ':.2f', 
            'risk_category': True,
            'latitude': False, 
            'longitude': False
        },
        color_continuous_scale='Viridis',
        title='Dengue Risk Scores by City',
        projection='natural earth'
    )
    
    fig.update_layout(
        title_x=0.5,
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
        )
    )
    
    return fig

def risk_correlations(df):
    """
    Create correlation heatmap of risk-related features
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing risk assessment data
    
    Returns:
    --------
    go.Figure
        Correlation heatmap
    """
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix, 
        title='Feature Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        labels=dict(x='Features', y='Features', color='Correlation')
    )
    
    fig.update_layout(
        title_x=0.5,
        width=1000,
        height=1000
    )
    
    return fig

def weather_impact_analysis(df):
    """
    Analyze weather impact on arbovirus risk
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing risk assessment and weather data
    
    Returns:
    --------
    go.Figure
        Scatter plot showing weather impact on arbovirus risk
    """
    # Choose key weather features to analyze
    weather_features = [
        'temperature_2m_max_mean', 
        'precipitation_sum_mean', 
        'wind_speed_10m_max_mean'
    ]
    
    # Create subplots
    fig = go.Figure()
    
    for feature in weather_features:
        fig.add_trace(
            go.Scatter(
                x=df[feature], 
                y=df['risk_score'],
                mode='markers',
                name=feature.replace('_mean', ''),
                marker=dict(
                    size=10,
                    opacity=0.7
                )
            )
        )
    
    fig.update_layout(
        title='Weather Features vs Dengue Risk Score',
        xaxis_title='Weather Feature Value',
        yaxis_title='Risk Score',
        legend_title='Weather Feature'
    )
    
    # Add trend line
    fig.add_trace(
        go.Scatter(
            x=df['temperature_2m_max_mean'], 
            y=np.poly1d(np.polyfit(df['temperature_2m_max_mean'], df['risk_score'], 1))(df['temperature_2m_max_mean']),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dot')
        )
    )
    
    return fig

def create_folium_risk_map(df):
    """
    Create an interactive Folium map with city risk markers
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing city-level risk data
    
    Returns:
    --------
    folium.Map
        Interactive risk map
    """
    # Create base map centered on mean coordinates
    m = folium.Map(
        location=[df['latitude'].mean(), df['longitude'].mean()], 
        zoom_start=5
    )
    
    # Create marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Color mapping for risk categories
    risk_colors = {
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red'
    }
    
    # Add markers for each city
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5 + (row['risk_score'] / 10),
            popup=f"""
            City: {row['city']}
            Risk Score: {row['risk_score']:.2f}
            Risk Category: {row['risk_category']}
            """,
            color=risk_colors.get(row['risk_category'], 'blue'),
            fill=True,
            fillColor=risk_colors.get(row['risk_category'], 'blue'),
            fillOpacity=0.7
        ).add_to(marker_cluster)
    
    return m

def generate_comprehensive_report(df):
    """
    Generate a comprehensive risk assessment report with visualizations
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing risk assessment data
    """
    # Create visualizations
    risk_choropleth = create_interactive_risk_choropleth(df)
    # temporal_heatmap = temporal_risk_heatmap(df)
    correlation_heatmap = risk_correlations(df)
    weather_impact = weather_impact_analysis(df)
    
    # Save visualizations
    pio.write_html(risk_choropleth, file='risk_choropleth.html')
    # pio.write_html(temporal_heatmap, file='temporal_heatmap.html')
    pio.write_html(correlation_heatmap, file='correlation_heatmap.html')
    pio.write_html(weather_impact, file='weather_impact.html')
    
    # Create Folium map
    risk_map = create_folium_risk_map(df)
    risk_map.save('interactive_risk_map.html')
    
    print("Comprehensive risk assessment report generated!")
    print("Output files:")
    print("- risk_choropleth.html")
    # print("- temporal_heatmap.html")
    print("- correlation_heatmap.html")
    print("- weather_impact.html")
    print("- interactive_risk_map.html")
