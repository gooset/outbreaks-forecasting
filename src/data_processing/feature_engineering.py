import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineering with input dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe for feature engineering
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
    
    def create_temporal_features(self):
        """
        Generate comprehensive temporal features
        
        Returns:
        --------
        FeatureEngineer
            Instance with added temporal features
        """
        # Basic temporal extraction
        self.df['month'] = self.df['date'].dt.month
        self.df['year'] = self.df['date'].dt.year
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        
        # Seasonal classification
        self.df['season'] = pd.cut(
            self.df['month'],
            bins=[0, 5, 10, 12], 
            labels=['Dry', 'Rainy', 'Dry'], 
            right=True, 
            ordered=False
        )
        
        # Cyclic encoding for temporal periodicity
        for col, max_val in [
            ('month', 12), 
            ('day_of_year', 365),
            ('week_of_year', 52), 
            ('day_of_week', 7)
        ]:
            self.df[f'{col}_sin'] = np.sin(2 * np.pi * self.df[col] / max_val)
            self.df[f'{col}_cos'] = np.cos(2 * np.pi * self.df[col] / max_val)
        
        return self
    
    def create_weather_features(self):
        """
        Generate advanced weather-related features
        
        Returns:
        --------
        FeatureEngineer
            Instance with added weather features
        """
        # Default weather columns
        weather_cols = [
            'temperature_2m_max', 'temperature_2m_min', 
            'precipitation_sum', 'wind_speed_10m_max'
        ]
        
        # Rolling window statistics
        windows = [3, 7, 14]
        for col in weather_cols:
            for window in windows:
                self.df[f'{col}_{window}d_mean'] = (
                    self.df.groupby('city')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
                self.df[f'{col}_{window}d_std'] = (
                    self.df.groupby('city')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).std())
                )
        
    # Disease-specific weather indicators
        self.df['optimal_mosquito_temp'] = (
            (self.df['temperature_2m_max'] >= 25) & 
            (self.df['temperature_2m_max'] <= 35)
        ).astype(int)
        
        self.df['breeding_conditions'] = (
            (self.df['precipitation_sum'] > 0) & 
            (self.df['temperature_2m_min'] > 20)
        ).astype(int)
        
        # Lag features for breeding conditions
        lag_days = [3, 7, 14]
        for lag in lag_days:
            self.df[f'breeding_conditions_lag_{lag}'] = (
                self.df.groupby('city')['breeding_conditions']
                .shift(lag).fillna(0)
            )

        # List of columns non-weather-related
        exclude_columns = [
            'latitude', 'longitude', 'country', 'country_id', 
            'status', 'occurrence_id', 'vector', 'source_type', 
            'location_type', 'city', 'date'
        ]

        all_weather_columns = [col for col in self.df.columns if col not in exclude_columns]

        # Apply forward fill and backward fill to the weather-related columns
        self.df[all_weather_columns] = self.df[all_weather_columns].fillna(method='ffill')
        self.df[all_weather_columns] = self.df[all_weather_columns].fillna(method='bfill')

        # Drop rows where there are still missing values in essential columns
        self.df.dropna(subset=exclude_columns, inplace=True)
        return self


