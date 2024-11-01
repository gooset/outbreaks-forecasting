#models/forecaster.py
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

class DengueProphetForecaster:
    def __init__(self, df):
        """
        Initialize Prophet forecasting for arbovirus risk.
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

    def prepare_data(self, target='arbovirus_bool'):
        """
        Prepare data aggregated by date and city for Prophet forecasting.
        """
        aggregated_data = self.df.groupby(['date', 'city']).agg(
            {
                target: 'sum',
                **{col: 'first' for col in self.df.columns if col not in ['date', 'city', target]}
            }
        ).reset_index()

        prophet_data = {
            city: city_df.rename(columns={'date': 'ds', target: 'y'})
            for city, city_df in aggregated_data.groupby('city')
        }
        return prophet_data

    def train_prophet_model(self, prophet_df, forecast_periods=90, add_regressors=True, regressors=None):
        """
        Train Prophet model for given dataframe with dynamic regressors.
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

    def generate_forecast_dataframe(self, forecast_periods=90, add_regressors=True):
        """
        Generate a DataFrame with forecasted values for all cities.
        """
        prophet_data = self.prepare_data()
        all_forecasts = []

        for city, city_data in prophet_data.items():
            model, city_forecast = self.train_prophet_model(
                city_data,
                forecast_periods=forecast_periods,
                add_regressors=add_regressors
            )

            forecast_df = city_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_df['city'] = city
            forecast_df.rename(columns={'ds': 'date', 'yhat': 'forecasted_arbovirus_cases'}, inplace=True)

            all_forecasts.append(forecast_df)

        forecast_df = pd.concat(all_forecasts, ignore_index=True)
        return forecast_df
