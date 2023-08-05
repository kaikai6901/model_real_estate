from utils.helpers import *
from unidecode import unidecode
import re
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
from settings import *
from preprocess.news.preprocess_params import *

class PreprocessNews:
    def __init__(self, client) -> None:
        self.client = client
    
    def get_projects(self):
        project_collection = self.client[DATABASE][PROJECT_COLLECTION]
        list_project = list(project_collection.find({}, {'project_id': 1, 'amenities': 1}))
        project_df = pd.DataFrame(list_project)
        project_df = extract_amenities(project_df)
        return project_df

    def get_commune(self):
        commune_colelction = self.client[DATABASE][COMMUNE_COLLECTION]
        response = list(commune_colelction.find({}, {'parser_response': 0}))

        list_commune = []
        for res in response:
            if res['commune'] is not None:
                commune = remove_special_character(res['commune'])
                if commune != '':
                    list_commune.append(res)
        commune_df = pd.DataFrame(list_commune)
        commune_df = extract_amenities(commune_df)

        district_df = commune_df.groupby(['district', 'province'])[amenities_features].mean()
        district_df.reset_index(inplace=True)
        district_df['formatted_address'] = district_df.apply(lambda x: ', '.join([unidecode(x['district']).strip().lower(), unidecode(x['province']).strip().lower()]), axis=1)
        full_commune_df = pd.concat([commune_df[district_df.columns], district_df])

        return full_commune_df

    def get_news_no_loc(self):
        news = get_news(self.client, have_project=False)
        news_df = pd.DataFrame(news)
        news_df.drop(columns=['_id'], inplace=True)
        news_df = extract_geocode(news_df)
        news_df['distance_to_center'] = news_df['loc'].apply(lambda x: calculate_distance(x))

        news_df['commune'] = news_df['commune'].apply(lambda x: get_formatted_string(x))
        news_df['district'] = news_df['district'].apply(lambda x: get_formatted_string(x))

        pop_density = get_population_feature(self.client)
        news_df['pop_density'] = news_df['district'].apply(lambda x: pop_density[unidecode(x).strip().lower()])

        label_encoding_cols = [
            'commune',
            'district'
        ]
    
        label_encoder = LabelEncoder()
        for col in label_encoding_cols:
            news_df[col + '_encode'] = label_encoder.fit_transform(news_df[col])

        news_df['source_encode'] = label_encoder.fit_transform(news_df['source'])

        return news_df
    
    def assign_news_has_project(self,news_has_project_df, project_df):
        news_has_project_df['project_id'] = news_has_project_df['base_project'].apply(lambda x: x['project_id'])
        news_has_project_df = news_has_project_df.merge(project_df, on='project_id')
        return news_has_project_df
    
    def assign_news_no_project(self, news_non_project_df, full_commune_df):
        news_non_project_df['formatted_address'] = news_non_project_df.apply(lambda x: get_formatted_address(x), axis=1)
        full_commune_df = full_commune_df.drop_duplicates()
        news_non_project_df = news_non_project_df.merge(full_commune_df, on='formatted_address')
        news_non_project_df.reset_index(inplace=True)
        return news_non_project_df
 
    def get_training_data(self):
        project_df = self.get_projects()
        full_commune_df = self.get_commune()
        news_df = self.get_news_no_loc()

        news_has_project_df = news_df[news_df['location_confidence'] == 1]
        news_non_project_df = news_df[news_df['location_confidence'] != 1]

        news_has_project_df = self.assign_news_has_project(news_has_project_df, project_df)
        news_non_project_df = self.assign_news_no_project(news_non_project_df, full_commune_df)

        full_news_df = pd.concat([news_has_project_df[list_cols], news_non_project_df[list_cols]])
        full_news_df.sort_values(['square', 'district_encode', 'distance_to_center', 'longitude', 'latitude'], inplace=True)
        full_news_df = full_news_df.ffill()
        full_news_df = full_news_df.bfill()

        return full_news_df
    

        



def get_training_data():
    client = get_mongodb_client()
    project_collection = client[DATABASE][PROJECT_COLLECTION]
    list_project = list(project_collection.find({}, {'project_id': 1, 'amenities': 1}))
    project_df = pd.DataFrame(list_project)

    project_df = extract_amenities(project_df=project_df)

    commune_collection = client[DATABASE][COMMUNE_COLLECTION]
    response = list(commune_collection.find({}, {'parser_response': 0}))

    list_commune = []
    for res in response:
        if res['commune'] is not None:
            commune = remove_special_character(res['commune'])
            if commune != '':
                list_commune.append(res)
    commune_df = pd.DataFrame(list_commune)
    commune_df = extract_amenities(commune_df)
      
