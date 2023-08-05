from datetime import datetime, timedelta
from utils.helpers import * 
from preprocess.project.preprocess_project import PreprocessProject
from train.project.training_params import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from urllib.parse import urlparse
import os

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae

class TrainingProject():
    def __init__(self, client) -> None:
        self.client = client
        self.preprocessor = PreprocessProject(client)
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.project_df = None
        self.full_df = None
        self.run_id = None
        self.mae = None
        self.predictions = None
        self.X_train = None
    
    def get_data(self):
        project_df, full_df = self.preprocessor.get_training_data()
        self.project_df = project_df
        full_df = full_df[list_cols]
        full_df.sort_values(['square', 'district_encode', 'longitude', 'latitude', 'distance_to_center'], inplace=True)
        full_df = full_df.ffill()
        full_df = full_df.bfill()
        self.full_df = full_df

    def train(self):
        X = self.full_df[features]
        y = self.full_df[label_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
        self.X_train = X_train
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        (rmse, mae) = eval_metrics(y_test, y_pred)
        self.mae = mae
        self.model.fit(X, y)
        self.predictions = self.model.predict(X)
    
    def store_mlflow(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            self.run_id = mlflow.active_run().info.run_id
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_metric("mae", self.mae)
            signature = infer_signature(self.X_train, self.predictions)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(
                    self.model, "model", registered_model_name="RFRegressor", signature=signature
                )
            else:
                mlflow.sklearn.log_model(self.model, "model", signature=signature)

    def store_features(self):
        self.project_df['run_id'] = self.run_id
        project_feature_collection = self.client['feature']['project']
        for id, row in self.project_df.iterrows():
            feature = {}
            feature['project_id'] = row['project_id']
            feature['name'] = row['name']
            feature['loc'] = row['loc']
            feature['feature'] = row[store_features].to_dict()
            feature['run_id'] = self.run_id
            feature['process_time'] = datetime.now()
            project_feature_collection.insert_one(feature)
