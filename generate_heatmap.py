#!/usr/bin/env python3
"""
Generate a correlation heatmap from the ATL Speech data CSV.
- Reads: /Users/anvitakallam/Documents/Atl Speech School/ATL Speech - Data Play - Sheet1.csv
- Outputs: heatmap PNG in the same directory
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "/Users/anvitakallam/Documents/Atl Speech School/ATL Speech - Data Play - Sheet1.csv"
OUTPUT_PATH = "/Users/anvitakallam/Documents/Atl Speech School/atl_speech_heatmap.png"


def main():
    df = pd.read_csv(CSV_PATH)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Coerce numeric columns
    for col in df.columns:
        if col.lower().startswith("audiologists"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Example: encode Presence of Audiology Program as binary for correlation if needed
    if "Presence of Audiology Program" in df.columns:
        df["Presence_Program_Binary"] = df["Presence of Audiology Program"].map({"Yes": 1, "No": 0})

    # Select numeric columns only
    num_df = df.select_dtypes(include=[np.number])

    if num_df.shape[1] == 0:
        print("No numeric columns found to correlate.")
        return

    corr = num_df.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)
    plt.title("Correlation Heatmap - ATL Speech")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved heatmap to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
