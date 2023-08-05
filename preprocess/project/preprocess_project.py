from utils.helpers import *
from unidecode import unidecode
import re
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
from settings import *


class PreprocessProject:
    def __init__(self, client) -> None:
        self.client = client
         
    def filter_project(self, list_project):
        vocabulary = set()
        try:
            db = self.client[DATABASE]
            commune_collection = db[COMMUNE_COLLECTION]
            list_commune = list(commune_collection.find({}, {'formatted_compound': 1}))

            for commune in list_commune:
                formatted_compound = commune['formatted_compound']
                if formatted_compound['district'] is not None and formatted_compound['district'] != '':
                    vocabulary.add(formatted_compound['district'])
                if formatted_compound['commune'] is not None and formatted_compound['commune'] != '':
                    vocabulary.add(formatted_compound['commune'])
            
            type_vocab = ['căn hộ chung cư', 'khu biệt thự', 'nhà phố', 'khu phức hợp', 'căn hộ dịch vụ', 'khu nghỉ dưỡng', 'cao ốc văn phòng', 'khu thương mại', 'khu dân cư', 'nhà ở xã hội', 'khu đô thị mới', 'khu tái định cư', 'khu đô thị', 'tòa nhà', 'chung cư mini', 'chung cư', 'mini', 'căn hộ']
            for word in type_vocab:
                vocabulary.add(unidecode(word).lower().strip())

            filtered_project = []
            for prj in tqdm(list_project):
                name = get_formatted_string(prj['name'])
                formatted_name = re.sub(special_character_pattern, '', name)
                if 'mini' in formatted_name:
                    continue
                for vocab in vocabulary:
                    formatted_name = formatted_name.replace(vocab, '')
                formatted_name = formatted_name.replace(' ', '')
                if formatted_name != '':
                    filtered_project.append(prj)
        except:
            return None
        return filtered_project


    def get_training_data(self):
        try:
            db = self.client[DATABASE]
            project_collection = db[PROJECT_COLLECTION]

            list_project = list(project_collection.find({}, {'parser_response': 0, 'amenities_detail': 0}))
            
            filtered_project = self.filter_project(list_project)
            print('filtered_project: ', len(filtered_project))
            project_df = pd.DataFrame(filtered_project)
            project_df = extract_amenities(project_df=project_df)

            project_df['distance_to_center'] = project_df['loc'].apply(lambda x: calculate_distance(x))
            project_df = extract_geocode(project_df)
            project_df['commune'] = project_df['address'].apply(lambda x: unidecode(x['compound']['commune']).strip().lower())
            project_df['district'] = project_df['address'].apply(lambda x: unidecode(x['compound']['district']).strip().lower())

            pop_density = get_population_feature(self.client)

            project_df['pop_density'] = project_df['district'].apply(lambda x: pop_density[unidecode(x).strip().lower()])

            print(len(project_df))
            label_encoding_cols = [
                'commune',
                'district'
            ]

            label_encoder = LabelEncoder()
            for col in label_encoding_cols:
                project_df[col + '_encode'] = label_encoder.fit_transform(project_df[col])
            project_df.drop(columns=['source'], inplace=True)

            news_df = get_news(self.client)
            print(len(news_df))
            
            list_project_id = list(set(news_df['project_id']))
            project_df = project_df[project_df['project_id'].isin(list_project_id)]

            full_df = project_df.merge(news_df[['project_id', 'source', 'price_per_m2', 'square', 'n_bedrooms']], on='project_id')

            full_df['source_encode'] = label_encoder.fit_transform(full_df['source'])
            full_df['square_change'] = full_df['square'] - full_df['avg_square']
            print(len(full_df))
            return project_df , full_df
        except:
            return None

            