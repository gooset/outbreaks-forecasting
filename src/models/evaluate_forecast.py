import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from prophet import Prophet

class DengueProphetEvaluator:
    def __init__(self, forecaster):
        """
        Initialize the evaluator with a DengueProphetForecaster instance.
        
        :param forecaster: DengueProphetForecaster instance
        """
        self.forecaster = forecaster
    
    def _train_prophet_model(self, prophet_df, forecast_periods=90, regressors=None):
        """
        Internal method to train a Prophet model with similar parameters to forecaster.
        
        :param prophet_df: DataFrame prepared for Prophet
        :param forecast_periods: Number of periods to forecast
        :param regressors: List of regressor columns
        :return: Trained model and forecast
        """
        model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10
        )
        
        if regressors is None:
            regressors = ['temperature_2m_max', 'precipitation_sum', 'wind_speed_10m_max']
        
        for regressor in regressors:
            if regressor in prophet_df.columns:
                model.add_regressor(regressor)
        
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=forecast_periods)
        for regressor in regressors:
            if regressor in prophet_df.columns:
                future[regressor] = prophet_df[regressor].iloc[-1]
        
        forecast = model.predict(future)
        return model, forecast
    
    def cross_validation(self, prophet_data, validation_periods=30, fold_count=3):
        """
        Perform time series cross-validation for each city's Prophet model.
        
        :param prophet_data: Dictionary of city-specific Prophet dataframes
        :param validation_periods: Number of periods to validate
        :param fold_count: Number of cross-validation folds
        :return: Dictionary of validation metrics for each city
        """
        validation_results = {}
        
        for city, city_data in prophet_data.items():
            city_metrics = {
                'mse': [],
                'mae': [],
                'rmse': [],
                'r2': [],
                'mape': []
            }
            
            for fold in range(fold_count):
                # Split data for training and validation
                train_end = len(city_data) - (fold + 1) * validation_periods
                train_data = city_data.iloc[:train_end]
                test_data = city_data.iloc[train_end:train_end + validation_periods]
                
                # Train model and forecast
                model, forecast = self._train_prophet_model(train_data, forecast_periods=validation_periods)
                
                # Extract forecast and actual values
                forecast_values = forecast['yhat'].values[:validation_periods]
                actual_values = test_data['y'].values
                
                # Calculate metrics
                city_metrics['mse'].append(mean_squared_error(actual_values, forecast_values))
                city_metrics['mae'].append(mean_absolute_error(actual_values, forecast_values))
                city_metrics['rmse'].append(np.sqrt(mean_squared_error(actual_values, forecast_values)))
                city_metrics['r2'].append(r2_score(actual_values, forecast_values))
                city_metrics['mape'].append(np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100)
            
            # Compute average metrics for the city
            validation_results[city] = {
                metric: np.mean(values) for metric, values in city_metrics.items()
            }
        
        return validation_results