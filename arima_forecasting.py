#!/usr/bin/env python3
"""
ARIMA Forecasting for Child Outcome Data
This script forecasts % meeting 1, % meeting 3, and % meeting 6 for 2023 and 2024
using ARIMA models with comprehensive evaluation and accuracy metrics.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import itertools
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ARIMAForecaster:
    def __init__(self, csv_path):
        """Initialize the forecaster with data from CSV file."""
        self.csv_path = csv_path
        self.df = None
        self.forecast_results = {}
        self.model_evaluations = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the data for forecasting."""
        print("Loading and preparing data...")
        
        # Load the CSV file
        self.df = pd.read_csv(self.csv_path)
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Convert percentage columns to numeric, handling null values
        percentage_cols = ['% meeting 1', '% meeting 3', '% meeting 6']
        for col in percentage_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Filter out rows with missing data for training (2018-2022)
        self.df_train = self.df[self.df['Year'].isin([2018, 2019, 2020, 2021, 2022])].copy()
        
        # Create forecast rows for 2023 and 2024
        self.df_forecast = self.df[self.df['Year'].isin([2023, 2024])].copy()
        
        print(f"Training data shape: {self.df_train.shape}")
        print(f"Forecast data shape: {self.df_forecast.shape}")
        
        return self.df_train, self.df_forecast
    
    def check_stationarity(self, series, title="Time Series"):
        """Check if a time series is stationary using Augmented Dickey-Fuller test."""
        # Check if series is constant
        if series.nunique() <= 1:
            print(f"\n{title} - Stationarity Test:")
            print("Series is constant (all values are the same)")
            print("Constant series are considered stationary")
            return True
        
        try:
            result = adfuller(series.dropna())
            print(f"\n{title} - Stationarity Test:")
            print(f"ADF Statistic: {result[0]:.4f}")
            print(f"p-value: {result[1]:.4f}")
            print(f"Critical Values:")
            for key, value in result[4].items():
                print(f"\t{key}: {value:.4f}")
            
            if result[1] <= 0.05:
                print("Series is stationary (p-value <= 0.05)")
                return True
            else:
                print("Series is non-stationary (p-value > 0.05)")
                return False
        except ValueError as e:
            print(f"\n{title} - Stationarity Test:")
            print(f"Error in stationarity test: {e}")
            print("Assuming series is stationary")
            return True
    
    def find_best_arima_params(self, series, max_p=3, max_d=2, max_q=3):
        """Find the best ARIMA parameters using AIC criterion."""
        # Handle constant series
        if pd.Series(series).nunique() <= 1:
            print("Series is constant, using simple ARIMA(0,0,0) model")
            try:
                model = ARIMA(series, order=(0, 0, 0))
                fitted_model = model.fit()
                return (0, 0, 0), fitted_model
            except:
                # If even (0,0,0) fails, return None
                return None, None
        
        best_aic = np.inf
        best_params = None
        best_model = None
        
        # Generate all possible parameter combinations
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)
        
        param_combinations = list(itertools.product(p_values, d_values, q_values))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for params in param_combinations:
            try:
                model = ARIMA(series, order=params)
                fitted_model = model.fit()
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_params = params
                    best_model = fitted_model
                    
            except Exception as e:
                continue
        
        print(f"Best ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
        return best_params, best_model
    
    def evaluate_model(self, model, test_data, metric_name):
        """Evaluate model performance using various metrics."""
        # Generate predictions
        predictions = model.forecast(steps=len(test_data))
        
        # Calculate metrics
        mae = mean_absolute_error(test_data, predictions)
        mse = mean_squared_error(test_data, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        r2 = r2_score(test_data, predictions)
        
        evaluation = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        print(f"\n{metric_name} Model Evaluation:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")
        
        return evaluation
    
    def forecast_for_state(self, state_name, metric_col):
        """Forecast a specific metric for a specific state."""
        print(f"\n{'='*60}")
        print(f"Forecasting {metric_col} for {state_name}")
        print(f"{'='*60}")
        
        # Get data for this state and metric
        state_data = self.df_train[
            (self.df_train['State (rank)'] == state_name) & 
            (self.df_train[metric_col].notna())
        ].copy()
        
        if len(state_data) < 3:
            print(f"Insufficient data for {state_name} - {metric_col}")
            return None
        
        # Sort by year
        state_data = state_data.sort_values('Year')
        series = state_data[metric_col].values
        
        print(f"Data points: {len(series)}")
        print(f"Years: {state_data['Year'].min()} - {state_data['Year'].max()}")
        print(f"Values: {series}")
        
        # Check stationarity
        is_stationary = self.check_stationarity(pd.Series(series), f"{state_name} - {metric_col}")
        
        # Find best ARIMA parameters
        best_params, best_model = self.find_best_arima_params(series)
        
        if best_model is None:
            print(f"Could not fit ARIMA model for {state_name} - {metric_col}")
            return None
        
        # Generate forecasts for 2023 and 2024
        forecast_steps = 2  # 2023 and 2024
        forecast = best_model.forecast(steps=forecast_steps)
        forecast_ci = best_model.get_forecast(steps=forecast_steps).conf_int()
        
        print(f"\nForecast for {state_name} - {metric_col}:")
        print(f"2023: {forecast[0]:.2f} (CI: {forecast_ci[0, 0]:.2f} - {forecast_ci[0, 1]:.2f})")
        print(f"2024: {forecast[1]:.2f} (CI: {forecast_ci[1, 0]:.2f} - {forecast_ci[1, 1]:.2f})")
        
        # Store results
        result = {
            'state': state_name,
            'metric': metric_col,
            'model': best_model,
            'params': best_params,
            'forecast_2023': forecast[0],
            'forecast_2024': forecast[1],
            'ci_2023_lower': forecast_ci[0, 0],
            'ci_2023_upper': forecast_ci[0, 1],
            'ci_2024_lower': forecast_ci[1, 0],
            'ci_2024_upper': forecast_ci[1, 1],
            'is_stationary': is_stationary
        }
        
        return result
    
    def run_forecasting(self):
        """Run forecasting for all states and metrics."""
        print("Starting ARIMA forecasting process...")
        
        # Get unique states with sufficient data
        states_with_data = self.df_train['State (rank)'].value_counts()
        states_to_forecast = states_with_data[states_with_data >= 3].index.tolist()
        
        print(f"States with sufficient data: {len(states_to_forecast)}")
        
        metrics = ['% meeting 1', '% meeting 3', '% meeting 6']
        
        all_forecasts = []
        
        for state in states_to_forecast:
            for metric in metrics:
                result = self.forecast_for_state(state, metric)
                if result:
                    all_forecasts.append(result)
        
        self.forecast_results = all_forecasts
        print(f"\nCompleted forecasting for {len(all_forecasts)} state-metric combinations")
        
        return all_forecasts
    
    def create_forecast_dataframe(self):
        """Create a comprehensive dataframe with all forecasts."""
        if not self.forecast_results:
            print("No forecast results available. Run forecasting first.")
            return None
        
        forecast_data = []
        
        for result in self.forecast_results:
            # 2023 forecast
            forecast_data.append({
                'State': result['state'],
                'Year': 2023,
                'Metric': result['metric'],
                'Forecasted_Value': result['forecast_2023'],
                'CI_Lower': result['ci_2023_lower'],
                'CI_Upper': result['ci_2023_upper'],
                'ARIMA_Params': str(result['params']),
                'Is_Stationary': result['is_stationary']
            })
            
            # 2024 forecast
            forecast_data.append({
                'State': result['state'],
                'Year': 2024,
                'Metric': result['metric'],
                'Forecasted_Value': result['forecast_2024'],
                'CI_Lower': result['ci_2024_lower'],
                'CI_Upper': result['ci_2024_upper'],
                'ARIMA_Params': str(result['params']),
                'Is_Stationary': result['is_stationary']
            })
        
        forecast_df = pd.DataFrame(forecast_data)
        return forecast_df
    
    def create_original_format_forecast(self):
        """Create forecast in the original CSV format."""
        if not self.forecast_results:
            print("No forecast results available. Run forecasting first.")
            return None
        
        # Create a copy of the original dataframe
        result_df = self.df.copy()
        
        # Fill in the forecast values
        for result in self.forecast_results:
            state = result['state']
            metric = result['metric']
            
            # Update 2023 values
            mask_2023 = (result_df['State (rank)'] == state) & (result_df['Year'] == 2023)
            result_df.loc[mask_2023, metric] = result['forecast_2023']
            
            # Update 2024 values
            mask_2024 = (result_df['State (rank)'] == state) & (result_df['Year'] == 2024)
            result_df.loc[mask_2024, metric] = result['forecast_2024']
        
        return result_df
    
    def generate_summary_statistics(self):
        """Generate summary statistics for the forecasts."""
        if not self.forecast_results:
            print("No forecast results available.")
            return None
        
        summary_data = []
        
        for metric in ['% meeting 1', '% meeting 3', '% meeting 6']:
            metric_forecasts = [r for r in self.forecast_results if r['metric'] == metric]
            
            if metric_forecasts:
                values_2023 = [r['forecast_2023'] for r in metric_forecasts]
                values_2024 = [r['forecast_2024'] for r in metric_forecasts]
                
                summary_data.append({
                    'Metric': metric,
                    'Year': 2023,
                    'Count': len(values_2023),
                    'Mean': np.mean(values_2023),
                    'Median': np.median(values_2023),
                    'Std': np.std(values_2023),
                    'Min': np.min(values_2023),
                    'Max': np.max(values_2023)
                })
                
                summary_data.append({
                    'Metric': metric,
                    'Year': 2024,
                    'Count': len(values_2024),
                    'Mean': np.mean(values_2024),
                    'Median': np.median(values_2024),
                    'Std': np.std(values_2024),
                    'Min': np.min(values_2024),
                    'Max': np.max(values_2024)
                })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def save_results(self, output_dir=None):
        """Save all results to CSV files."""
        if output_dir is None:
            output_dir = os.path.dirname(self.csv_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed forecast results
        forecast_df = self.create_forecast_dataframe()
        if forecast_df is not None:
            forecast_file = os.path.join(output_dir, f"arima_forecasts_detailed_{timestamp}.csv")
            forecast_df.to_csv(forecast_file, index=False)
            print(f"Detailed forecasts saved to: {forecast_file}")
        
        # Save forecast in original format
        original_format_df = self.create_original_format_forecast()
        if original_format_df is not None:
            original_file = os.path.join(output_dir, f"arima_forecasts_original_format_{timestamp}.csv")
            original_format_df.to_csv(original_file, index=False)
            print(f"Original format forecasts saved to: {original_file}")
        
        # Save summary statistics
        summary_df = self.generate_summary_statistics()
        if summary_df is not None:
            summary_file = os.path.join(output_dir, f"arima_forecast_summary_{timestamp}.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"Summary statistics saved to: {summary_file}")
        
        return {
            'detailed_forecasts': forecast_file if forecast_df is not None else None,
            'original_format': original_file if original_format_df is not None else None,
            'summary': summary_file if summary_df is not None else None
        }

def main():
    """Main function to run the ARIMA forecasting."""
    # Path to the CSV file
    csv_path = "/Users/anvitakallam/Documents/Atl Speech School/180DC Reseach Sheet - Child Outcome - Forecast.csv"
    
    # Initialize the forecaster
    forecaster = ARIMAForecaster(csv_path)
    
    # Load and prepare data
    df_train, df_forecast = forecaster.load_and_prepare_data()
    
    # Run forecasting
    forecast_results = forecaster.run_forecasting()
    
    # Save results
    output_files = forecaster.save_results()
    
    print("\n" + "="*80)
    print("ARIMA FORECASTING COMPLETED")
    print("="*80)
    print(f"Total forecasts generated: {len(forecast_results)}")
    print(f"Output files created:")
    for file_type, file_path in output_files.items():
        if file_path:
            print(f"  - {file_type}: {file_path}")
    
    # Display sample results
    if forecast_results:
        print("\nSample forecast results:")
        sample_results = forecast_results[:5]  # Show first 5 results
        for result in sample_results:
            print(f"{result['state']} - {result['metric']}:")
            print(f"  2023: {result['forecast_2023']:.2f}")
            print(f"  2024: {result['forecast_2024']:.2f}")
            print(f"  ARIMA params: {result['params']}")
            print()

if __name__ == "__main__":
    main()
