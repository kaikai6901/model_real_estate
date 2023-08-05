from utils.helpers import *
from train.news.training_news import *

def process():
    client = get_mongodb_client()
    training = TrainingNews(client)
    training.get_data()
    training.train()
    training.save_scores()
process()
