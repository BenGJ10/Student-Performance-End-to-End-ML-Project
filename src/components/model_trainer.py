import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """
    This module trains and evaluates multiple regression models using training and testing datasets.
    It compares the models based on their R² score and saves the best-performing model to disk.

    Classes:

    ModelTrainerConfig : Defines configuration settings for model training, including the model save path.

    ModelTrainer : Handles training of multiple regression models, evaluates their performance, and saves the best model.

    Functions:

      initiate_model_trainer:

    Splits the input data, trains all models, evaluates them using R² score,
    identifies the best model, saves it, and returns its R² score.
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Training and Testing input data")
            X_train, Y_train, X_test, Y_test = (
                train_array[:, :-1],
                train_array[:, -1], 
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Catboosting": CatBoostRegressor(verbose = False),
                "XGBoosting": XGBRegressor(),
                "Adaboosting": AdaBoostRegressor(),
                "K-Neighbors": KNeighborsRegressor()
            } 

            model_report: dict = evaluate_model(X_train, Y_train, X_test, Y_test, models = models) 

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(Y_test, predicted)
            return r2_square
        

        except Exception as e:
            raise CustomException(e, sys) 
