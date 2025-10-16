#!/usr/bin/env python3
"""
Generate US choropleth heatmaps for child outcome data by year (2018-2022).
- Creates individual heatmaps for each metric and year
- Uses unified color scale across all years and metrics for direct comparison
- Outputs HTML and PNG files for each combination
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
        'title_prefix': 'Percentage Meeting Outcome 1',
        'filename_prefix': 'outcome1',
        'color_scale': 'Viridis'
    },
    '% meeting 3': {
        'title_prefix': 'Percentage Meeting Outcome 3',
        'filename_prefix': 'outcome3',
        'color_scale': 'Viridis'
    },
    '% meeting 6': {
        'title_prefix': 'Percentage Meeting Outcome 6',
        'filename_prefix': 'outcome6',
        'color_scale': 'Viridis'
    }
}

YEARS = [2018, 2019, 2020, 2021, 2022]


def get_unified_scale_range(df):
    """Get the unified scale range across all metrics and years."""
    min_val = float('inf')
    max_val = float('-inf')
    
    for year in YEARS:
        df_year = df[df['Year'] == year].copy()
        df_year = df_year[df_year['State (rank)'].notna()].copy()
        
        for metric in METRICS.keys():
            df_year[metric] = pd.to_numeric(df_year[metric], errors='coerce')
            metric_data = df_year[df_year[metric].notna()][metric]
            if len(metric_data) > 0:
                min_val = min(min_val, metric_data.min())
                max_val = max(max_val, metric_data.max())
    
    return min_val, max_val


def create_heatmap_for_metric_year(df, metric, metric_info, year, color_range):
    """Create a single choropleth map for a specific metric and year."""
    
    # Filter for specific year and clean
    df_year = df[df['Year'] == year].copy()
    df_year = df_year[df_year['State (rank)'].notna()].copy()
    
    # Map to state abbreviations
    df_year['state_abbr'] = df_year['State (rank)'].map(US_STATE_ABBR)
    df_year = df_year[df_year['state_abbr'].notna()].copy()
    
    # Convert metric to numeric
    df_year[metric] = pd.to_numeric(df_year[metric], errors='coerce')
    
    # Remove rows with missing values for this metric
    df_metric = df_year[df_year[metric].notna()].copy()
    
    if len(df_metric) == 0:
        print(f"No data available for {metric} in {year}")
        return None
    
    # Create choropleth with reversed color scale and unified range
    fig = px.choropleth(
        df_metric,
        locations="state_abbr",
        locationmode="USA-states",
        color=metric,
        color_continuous_scale=metric_info['color_scale'] + '_r',
        scope="usa",
        hover_name="State (rank)",
        hover_data={metric: ':.1f'},
        title=f"{metric_info['title_prefix']} ({year})",
        labels={metric: 'Percentage'},
        range_color=color_range
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        font=dict(size=14)
    )
    
    return fig


def create_combined_year_heatmap(df, year, color_range):
    """Create a combined subplot showing all three metrics for a specific year."""
    
    # Filter for specific year
    df_year = df[df['Year'] == year].copy()
    df_year = df_year[df_year['State (rank)'].notna()].copy()
    df_year['state_abbr'] = df_year['State (rank)'].map(US_STATE_ABBR)
    df_year = df_year[df_year['state_abbr'].notna()].copy()
    
    # Convert metrics to numeric
    for metric in METRICS.keys():
        df_year[metric] = pd.to_numeric(df_year[metric], errors='coerce')
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(METRICS.keys()),
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}, {"type": "choropleth"}]]
    )
    
    # Add each metric as a subplot with unified color range
    for i, (metric, metric_info) in enumerate(METRICS.items(), 1):
        df_metric = df_year[df_year[metric].notna()].copy()
        
        if len(df_metric) > 0:
            fig.add_trace(
                go.Choropleth(
                    locations=df_metric['state_abbr'],
                    z=df_metric[metric],
                    locationmode='USA-states',
                    colorscale=metric_info['color_scale'] + '_r',
                    zmin=color_range[0],
                    zmax=color_range[1],
                    showscale=True if i == 3 else False,
                    colorbar=dict(x=1.02) if i == 3 else None,
                    hovertemplate=f"<b>%{{location}}</b><br>{metric}: %{{z:.1f}}%<extra></extra>"
                ),
                row=1, col=i
            )
    
    # Update layout
    fig.update_layout(
        title_text=f"2022 Child Outcome Metrics by State ({year})",
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
    
    # Get unified color scale range across all metrics and years
    print("Calculating unified color scale range across all years and metrics...")
    color_range = get_unified_scale_range(df)
    print(f"Unified color range: {color_range[0]:.1f} - {color_range[1]:.1f}")
    
    # Create individual heatmaps for each metric and year
    print("Creating individual heatmaps...")
    
    for year in YEARS:
        print(f"\nProcessing year {year}...")
        
        for metric, metric_info in METRICS.items():
            print(f"  Creating heatmap for {metric} in {year}...")
            fig = create_heatmap_for_metric_year(df, metric, metric_info, year, color_range)
            
            if fig:
                # Save HTML
                html_path = os.path.join(OUTPUT_DIR, f"{metric_info['filename_prefix']}_{year}_heatmap.html")
                fig.write_html(html_path, include_plotlyjs="cdn")
                print(f"    Saved HTML: {html_path}")
                
                # Save PNG
                png_path = os.path.join(OUTPUT_DIR, f"{metric_info['filename_prefix']}_{year}_heatmap.png")
                try:
                    fig.write_image(png_path, scale=2, width=1000, height=600)
                    print(f"    Saved PNG: {png_path}")
                except Exception as e:
                    print(f"    PNG save failed for {metric} {year}: {e}")
    
    # Create combined heatmaps for each year
    print("\nCreating combined yearly heatmaps...")
    
    for year in YEARS:
        print(f"Creating combined heatmap for {year}...")
        combined_fig = create_combined_year_heatmap(df, year, color_range)
        
        if combined_fig:
            # Save combined HTML
            combined_html = os.path.join(OUTPUT_DIR, f"all_outcomes_{year}_heatmap.html")
            combined_fig.write_html(combined_html, include_plotlyjs="cdn")
            print(f"  Saved combined HTML: {combined_html}")
            
            # Save combined PNG
            combined_png = os.path.join(OUTPUT_DIR, f"all_outcomes_{year}_heatmap.png")
            try:
                combined_fig.write_image(combined_png, scale=2, width=1500, height=600)
                print(f"  Saved combined PNG: {combined_png}")
            except Exception as e:
                print(f"  Combined PNG save failed for {year}: {e}")
    
    print("\nYearly heatmap generation completed!")


if __name__ == "__main__":
    main()
