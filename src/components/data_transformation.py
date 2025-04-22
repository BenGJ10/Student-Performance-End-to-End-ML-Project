# Take whatever output we are getting from ingestion module and apply various transformations.
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    """
    Data Transformation component is responsible for transforming the data into the required format.\n
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Creating all pickle files for data transformation
    def get_data_transformer_object(self):
        """
        This function is responsible for creating the data transformation pipeline.\n
        It includes the following steps:
        1. Numerical columns are imputed with median and scaled using StandardScaler.
        2. Categorical columns are imputed with the most frequent value, one-hot encoded, and scaled using StandardScaler.
        3. The preprocessor object is created using ColumnTransformer to apply the pipelines to the respective columns.
        4. The preprocessor object is returned.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "lunch",
                "parental_level_of_education",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps = [
                ("imputer", SimpleImputer(strategy = "median")),
                ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("onehotencoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Numerical columns {numerical_columns}")
            logging.info(f"Categorical columns {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, numerical_columns),
                    ("categorical_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException (e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Reading train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columsn = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columsn = [target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columsn = [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]
        except:
            pass
