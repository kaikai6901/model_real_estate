from utils.helpers import *
from train.project.training_project import *

def process():
    client = get_mongodb_client()
    training = TrainingProject(client)
    training.get_data()
    training.train()
    training.store_mlflow()
    training.store_features()
    client.close()

process()
