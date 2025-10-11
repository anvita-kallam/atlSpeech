#!/usr/bin/env python3
"""
Generate US choropleth heatmaps for 2022 child outcome data.
- Creates separate maps for % meeting 1, % meeting 3, and % meeting 6
- Uses data from the original CSV for 2022 values
- Outputs HTML and PNG files for each metric
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CSV_PATH = "/Users/anvitakallam/Documents/Atl Speech School/180DC Reseach Sheet - Child Outcome - Forecast.csv"
OUTPUT_DIR = "/Users/anvitakallam/Documents/Atl Speech School"

US_STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

METRICS = {
    '% meeting 1': {
        'title': 'Percentage Meeting Outcome 1 (2022)',
        'color_scale': 'Viridis',
        'filename': 'outcome1_2022_heatmap'
    },
    '% meeting 3': {
        'title': 'Percentage Meeting Outcome 3 (2022)', 
        'color_scale': 'Viridis',
        'filename': 'outcome3_2022_heatmap'
    },
    '% meeting 6': {
        'title': 'Percentage Meeting Outcome 6 (2022)',
        'color_scale': 'Viridis',
        'filename': 'outcome6_2022_heatmap'
    }
}


def create_heatmap_for_metric(df, metric, metric_info):
    """Create a single choropleth map for a specific metric."""
    
    # Filter for 2022 data and clean
    df_2022 = df[df['Year'] == 2022].copy()
    df_2022 = df_2022[df_2022['State (rank)'].notna()].copy()
    
    # Map to state abbreviations
    df_2022['state_abbr'] = df_2022['State (rank)'].map(US_STATE_ABBR)
    df_2022 = df_2022[df_2022['state_abbr'].notna()].copy()
    
    # Convert metric to numeric
    df_2022[metric] = pd.to_numeric(df_2022[metric], errors='coerce')
    
    # Remove rows with missing values for this metric
    df_metric = df_2022[df_2022[metric].notna()].copy()
    
    if len(df_metric) == 0:
        print(f"No data available for {metric} in 2022")
        return None
    
    # Create choropleth
    fig = px.choropleth(
        df_metric,
        locations="state_abbr",
        locationmode="USA-states",
        color=metric,
        color_continuous_scale=metric_info['color_scale'],
        scope="usa",
        hover_name="State (rank)",
        hover_data={metric: ':.1f'},
        title=metric_info['title'],
        labels={metric: 'Percentage'}
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        font=dict(size=14)
    )
    
    return fig


def create_combined_heatmap(df):
    """Create a combined subplot showing all three metrics."""
    
    # Filter for 2022 data
    df_2022 = df[df['Year'] == 2022].copy()
    df_2022 = df_2022[df_2022['State (rank)'].notna()].copy()
    df_2022['state_abbr'] = df_2022['State (rank)'].map(US_STATE_ABBR)
    df_2022 = df_2022[df_2022['state_abbr'].notna()].copy()
    
    # Convert metrics to numeric
    for metric in METRICS.keys():
        df_2022[metric] = pd.to_numeric(df_2022[metric], errors='coerce')
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(METRICS.keys()),
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}, {"type": "choropleth"}]]
    )
    
    # Add each metric as a subplot
    for i, (metric, metric_info) in enumerate(METRICS.items(), 1):
        df_metric = df_2022[df_2022[metric].notna()].copy()
        
        if len(df_metric) > 0:
            fig.add_trace(
                go.Choropleth(
                    locations=df_metric['state_abbr'],
                    z=df_metric[metric],
                    locationmode='USA-states',
                    colorscale=metric_info['color_scale'],
                    showscale=True if i == 3 else False,
                    colorbar=dict(x=1.02) if i == 3 else None,
                    hovertemplate=f"<b>%{{location}}</b><br>{metric}: %{{z:.1f}}%<extra></extra>"
                ),
                row=1, col=i
            )
    
    # Update layout
    fig.update_layout(
        title_text="2022 Child Outcome Metrics by State",
        title_x=0.5,
        geo=dict(scope='usa'),
        margin=dict(l=0, r=0, t=80, b=0),
        font=dict(size=12)
    )
    
    return fig


def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    print("Creating individual heatmaps...")
    
    # Create individual heatmaps for each metric
    for metric, metric_info in METRICS.items():
        print(f"Creating heatmap for {metric}...")
        fig = create_heatmap_for_metric(df, metric, metric_info)
        
        if fig:
            # Save HTML
            html_path = os.path.join(OUTPUT_DIR, f"{metric_info['filename']}.html")
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"Saved HTML: {html_path}")
            
            # Save PNG
            png_path = os.path.join(OUTPUT_DIR, f"{metric_info['filename']}.png")
            try:
                fig.write_image(png_path, scale=2, width=1000, height=600)
                print(f"Saved PNG: {png_path}")
            except Exception as e:
                print(f"PNG save failed for {metric}: {e}")
    
    print("Creating combined heatmap...")
    
    # Create combined heatmap
    combined_fig = create_combined_heatmap(df)
    
    if combined_fig:
        # Save combined HTML
        combined_html = os.path.join(OUTPUT_DIR, "all_outcomes_2022_heatmap.html")
        combined_fig.write_html(combined_html, include_plotlyjs="cdn")
        print(f"Saved combined HTML: {combined_html}")
        
        # Save combined PNG
        combined_png = os.path.join(OUTPUT_DIR, "all_outcomes_2022_heatmap.png")
        try:
            combined_fig.write_image(combined_png, scale=2, width=1500, height=600)
            print(f"Saved combined PNG: {combined_png}")
        except Exception as e:
            print(f"Combined PNG save failed: {e}")
    
    print("Heatmap generation completed!")


if __name__ == "__main__":
    main()
