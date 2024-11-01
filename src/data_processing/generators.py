import random
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

faker = Faker()
random.seed(42)
faker.seed_instance(42)

def generate_arbovirus_data(cities_info: pd.DataFrame, start_date: str = '2019-01-01', 
                        end_date: str = '2023-12-31', target_records: int = 20000) -> pd.DataFrame:
    """Generate synthetic arbovirus occurrence data."""
    arbovirus_data = []
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    records_per_city = max(1, target_records // len(cities_info))

    for _, row in cities_info.iterrows():
        city = row['city']
        population = row['population']
        arbovirus_case_prob = min(max(0.20 + (1_000_000 / population) * 0.05, 0.20), 0.35)
        
        for _ in range(records_per_city):
            date = faker.date_between_dates(date_start=start, date_end=end)
            arbovirus_bool = int(random.random() < arbovirus_case_prob)
            arbovirus_data.append([city, date, arbovirus_bool])

    return pd.DataFrame(arbovirus_data, columns=['city', 'date', 'arbovirus_bool'])

def generate_vector_sightings(cities_info: pd.DataFrame, start_year: int = 2019, 
                            end_year: int = 2023, target_records: int = 500) -> pd.DataFrame:
    """Generate synthetic vector sighting data."""
    vector_data = []
    source_types = ["published", "unpublished", "survey", "literature", "museum_specimen"]
    species_list = [
        "Aedes aegypti", "Aedes africanus", "Anopheles gambiae",
        "Anopheles arabiensis", "Anopheles funestus", "Culex quinquefasciatus"
    ]
    
    for _, row in cities_info.iterrows():
        sightings_count = random.randint(3, 8)
        for _ in range(sightings_count):
            vector_data.append([
                faker.unique.random_int(),
                random.choice(species_list),
                random.choice(source_types),
                random.choice(['urban', 'rural', 'forest', 'wetland']),
                row['latitude'],
                row['longitude'],
                random.randint(start_year, end_year),
                row['city'],
                'Guinea',
                'GIN',
                'present' if random.random() < 0.7 else 'absent'
            ])
            
    return pd.DataFrame(vector_data, columns=[
        'occurrence_id', 'vector', 'source_type', 'location_type', 'latitude', 
        'longitude', 'year', 'city', 'country', 'country_id', 'status'
    ])