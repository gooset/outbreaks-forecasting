from sqlalchemy import Column, Float, Integer, String, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class City(Base):
    __tablename__ = 'cities'
    
    city = Column(String, primary_key=True)
    latitude = Column(Float)
    longitude = Column(Float)
    country = Column(String)
    population = Column(Integer)

class WeatherData(Base):
    __tablename__ = 'weather_data'
    
    city = Column(String, ForeignKey('cities.city'), primary_key=True)
    date = Column(Date, primary_key=True)
    temperature_2m_max = Column(Float)
    temperature_2m_min = Column(Float)
    precipitation_sum = Column(Float)
    wind_speed_10m_max = Column(Float)
    wind_gusts_10m_max = Column(Float)

class DengueData(Base):
    __tablename__ = 'arbovirus_data'
    
    city = Column(String, ForeignKey('cities.city'), primary_key=True)
    date = Column(Date, primary_key=True)
    arbovirus_bool = Column(Integer)

class VectorSighting(Base):
    __tablename__ = 'vector_sightings'
    
    occurrence_id = Column(Integer, primary_key=True)
    vector = Column(String)
    source_type = Column(String)
    location_type = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    year = Column(Integer)
    city = Column(String, ForeignKey('cities.city'))
    country = Column(String)
    country_id = Column(String)
    status = Column(String)