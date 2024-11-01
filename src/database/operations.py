from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
from datetime import datetime
from typing import Optional

from .models import Base, City, WeatherData, DengueData, VectorSighting

class DatabaseOperations:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
    def initialize_database(self):
        """Initialize database tables."""
        Base.metadata.create_all(self.engine)
        
    def insert_cities(self, cities_df: pd.DataFrame):
        """Insert cities data into database."""
        with self.Session() as session:
            for _, row in cities_df.iterrows():
                city = City(
                    city=row['city'],
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    country=row['country'],
                    population=row['population']
                )
                session.merge(city)
            session.commit()
            
    def insert_weather_data(self, weather_df: pd.DataFrame):
        """Insert weather data into database."""
        with self.Session() as session:
            for _, row in weather_df.iterrows():
                weather = WeatherData(
                    city=row['city'],
                    date=row['date'],
                    temperature_2m_max=row['temperature_2m_max'],
                    temperature_2m_min=row['temperature_2m_min'],
                    precipitation_sum=row['precipitation_sum'],
                    wind_speed_10m_max=row['wind_speed_10m_max'],
                    wind_gusts_10m_max=row['wind_gusts_10m_max']
                )
                session.merge(weather)
            session.commit()
            
    def insert_arbovirus_data(self, arbovirus_df: pd.DataFrame):
        """Insert arbovirus data into database."""
        with self.Session() as session:
            for _, row in arbovirus_df.iterrows():
                arbovirus = DengueData(
                    city=row['city'],
                    date=row['date'],
                    arbovirus_bool=row['arbovirus_bool']
                )
                session.merge(arbovirus)
            session.commit()
            
    def insert_vector_sightings(self, vector_df: pd.DataFrame):
        """Insert vector sightings data into database."""
        with self.Session() as session:
            for _, row in vector_df.iterrows():
                vector = VectorSighting(
                    occurrence_id=row['occurrence_id'],
                    vector=row['vector'],
                    source_type=row['source_type'],
                    location_type=row['location_type'],
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    year=row['year'],
                    city=row['city'],
                    country=row['country'],
                    country_id=row['country_id'],
                    status=row['status']
                )
                session.merge(vector)
            session.commit()
            
    def get_merged_data(self, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve and merge all data from database."""
        query = """
        SELECT 
            w.city, w.date, w.temperature_2m_max, w.temperature_2m_min,
            w.precipitation_sum, w.wind_speed_10m_max,
            COALESCE(d.arbovirus_bool, 0) as arbovirus_bool,
            v.occurrence_id, v.vector, v.source_type, v.location_type,
            v.latitude, v.longitude, v.country, v.country_id, v.status
        FROM weather_data w
        LEFT JOIN arbovirus_data d ON w.city = d.city AND w.date = d.date
        LEFT JOIN vector_sightings v ON w.city = v.city 
            AND EXTRACT(YEAR FROM w.date) = v.year
        """
        
        if start_date and end_date:
            query += f" WHERE w.date BETWEEN '{start_date}' AND '{end_date}'"
            
        return pd.read_sql(query, self.engine)
