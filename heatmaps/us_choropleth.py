#!/usr/bin/env python3
"""
Generate a US choropleth (state-colored heatmap) from the ATL Speech data CSV.
- Reads: /Users/anvitakallam/Documents/Atl Speech School/ATL Speech - Data Play - Sheet1.csv
- Colors states by: Audiologists per 100k population
- Outputs: HTML and PNG map in the same directory
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px

CSV_PATH = "/Users/anvitakallam/Documents/Atl Speech School/ATL Speech - Data Play - Sheet1.csv"
OUT_HTML = "/Users/anvitakallam/Documents/Atl Speech School/atl_speech_us_heatmap.html"
OUT_PNG = "/Users/anvitakallam/Documents/Atl Speech School/atl_speech_us_heatmap.png"

STATE_NAME_FIXES = {
    "Noebraska": "Nebraska",
    "Noevada": "Nevada",
    "Noew Hampshire": "New Hampshire",
    "Noew Jersey": "New Jersey",
    "Noew Mexico": "New Mexico",
    "Noew York": "New York",
}

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

EXCLUDE_NON_STATES = {
    "American Samoa", "Guam", "Marshall Islands", "Micronesia", "Northern Marianas",
    "Palau", "Puerto Rico", "Virgin Islands"
}


def clean_state_names(name: str) -> str:
    if pd.isna(name):
        return name
    name = name.strip()
    if name in STATE_NAME_FIXES:
        return STATE_NAME_FIXES[name]
    return name


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    state_col = [c for c in df.columns if c.lower().startswith("state")][0]
    metric_col = "Audiologists per 100k population"

    df[state_col] = df[state_col].apply(clean_state_names)

    # Filter to US states only
    df = df[~df[state_col].isin(EXCLUDE_NON_STATES)].copy()

    # Map to state abbreviations; drop rows we cannot map
    df["state_abbr"] = df[state_col].map(US_STATE_ABBR)
    df = df[~df["state_abbr"].isna()].copy()

    # Coerce metric to numeric
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    # Build choropleth
    fig = px.choropleth(
        df,
        locations="state_abbr",
        locationmode="USA-states",
        color=metric_col,
        color_continuous_scale="Viridis",
        scope="usa",
        hover_name=state_col,
        title="Audiologists per 100k population by State",
    )

    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))

    # Save HTML and PNG
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    try:
        fig.write_image(OUT_PNG, scale=2)
    except Exception:
        # PNG export requires kaleido; if missing, we still have HTML
        pass

    print(f"Saved HTML map to: {OUT_HTML}")
    if os.path.exists(OUT_PNG):
        print(f"Saved PNG map to: {OUT_PNG}")
    else:
        print("PNG export skipped (kaleido not installed). HTML map is available.")


if __name__ == "__main__":
    main()

