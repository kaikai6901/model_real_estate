import pymongo
import configparser
import re
import json
import urllib
from unidecode import unidecode
import math
from datetime import datetime, timedelta
import pandas as pd
from settings import *

special_character_pattern = "[" + re.escape("!@#$%^&*()-_+={}[]|\\/:;\"',.?`~") + "]"

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

def get_mongodb_client():
    host = config.get(MONGODB_CONFIG_NAME, 'host')
    user = config.get(MONGODB_CONFIG_NAME, 'user')
    password = config.get(MONGODB_CONFIG_NAME, 'password')
    mongo_uri = f'mongodb+srv://{user}:{password}@{host}/?retryWrites=true&w=majority'
    return pymongo.MongoClient(mongo_uri, retryWrites=False)

def get_formatted_string(s: str):
    return unidecode(s).strip().lower()

center_point = {
    'type': 'Point',
    'coordinates': [
        105.8549172,
        21.0234631
    ]
}

def calculate_distance(locations, target_location=center_point):
    def distance(lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        # Radius of the Earth in kilometers
        radius = 6371

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = radius * c
        return distance
    if isinstance(locations, list):
        total_distance = 0
        for location in locations:
            lon1, lat1 = location['coordinates']
            lon2, lat2 = target_location['coordinates']
            total_distance += distance(lat1, lon1, lat2, lon2)

        average_distance = total_distance / len(locations)
        return average_distance
    else:
        lon1, lat1 = locations['coordinates']
        lon2, lat2 = target_location['coordinates']
        return distance(lat1, lon1, lat2, lon2)

def get_population_feature(client):
    population_density_collection = client[FEATURE_DATABASE][POPULATION_COLLECTION]
    pop_density = dict()
    for density in population_density_collection.find({}):
        pop_density[density['formatted_district']] = density['density']
    
    return pop_density

def extract_amenities(project_df):
    original_name = ['primary_highway', 'secondary_highway', 'tertiary_highway', 'residential_highway', 'bus_stop', 'supermarket', 'mall', 'hospital', 'college', 'school', 'university']
    distances_key = [
        'amenities_in_500',
        'amenities_in_1000',
        'amenities_in_3000'
    ]

    def extract_amenities_field(row, field_name, original_field):
        return row[original_field].get(field_name, None)
    
    for field_name in distances_key:
        project_df[field_name] = project_df.apply(extract_amenities_field, args=[field_name, 'amenities'], axis=1)
    
    project_df.drop(columns=['amenities'], inplace=True)

    list_amenities_field = []
    for field_name in original_name:
        for distance_name in distances_key:
            new_field = '_'.join([field_name, distance_name.split('_')[-1]])
            list_amenities_field.append(new_field)
            project_df[new_field] = project_df.apply(extract_amenities_field, args=[field_name, distance_name, ], axis=1)

    project_df.drop(columns=distances_key, inplace=True)

    return project_df


def get_news(client, have_project=True):
    print('get_news')
    news_collection = client[DATABASE][NEWS_COLLECTION]
    oneMonthAgo = datetime.now() - timedelta(days=31)
    query = {'last_time_in_page': {'$gt': oneMonthAgo}}
    if have_project:
        query['base_project'] =  {'$exists': True}
    list_news = list(news_collection.find(query))
    news_df = pd.DataFrame(list_news)
    print(len(news_df))
    if have_project:
        news_df['project_id'] = news_df['base_project'].apply(lambda x: x['project_id']) 

    return news_df

def remove_special_character(s: str):
    return re.sub(special_character_pattern, '', s)

def extract_geocode(df):
    df['longitude'] = df['loc'].apply(lambda x: x['coordinates'][0])
    df['latitude'] = df['loc'].apply(lambda x: x['coordinates'][1])
    return df

def get_formatted_address(row):
    address = []
    if row['commune'] is not None:
        commune = remove_special_character(row['commune'])
        if commune != '':
            address.append(unidecode(row['commune']).strip().lower())
    address.append(unidecode(row['district']).strip().lower())
    address.append(unidecode(row['province']).strip().lower())
    return ', '.join(address)