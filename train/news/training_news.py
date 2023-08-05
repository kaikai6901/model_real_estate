from utils.helpers import *
from preprocess.news.preprocess_news import PreprocessNews
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from train.news.training_params import *
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae

class TrainingNews():
    def __init__(self, client) -> None:
        self.client = client
        self.preprocessor = PreprocessNews(client)
        self.full_news_df = None
        self.X = None
        self.y = None
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.scores = None
    def get_data(self):
        full_news_df = self.preprocessor.get_training_data()
        full_news_df.set_index('news_id', inplace=True)
        self.full_news_df = full_news_df
        self.X = full_news_df[features]
        self.y = full_news_df[label_name]
    
    def train(self):
        self.model.fit(self.X, self.y)
        self.scores = self.model.predict(self.X)
    
    def save_scores(self):
        news_collection = self.client[DATABASE][NEWS_COLLECTION]
        self.scores = (self.scores - self.y) / self.y * 100

        for index, value in self.scores.items():
            news_collection.update_one({'news_id': index}, {'$set': {'score': value}})
            
