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
from src.utils import save_object

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
                    ("scaler", StandardScaler(with_mean = False))
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
        """
        This method is responsible for performing data transformation on the training and testing datasets.\n
        Steps:
        1. Reads the training and testing CSV files into Pandas DataFrames.
        2. Initializes and retrieves the preprocessing object (typically includes scaling, encoding, etc.).
        3. Separates the input features and target column from both training and testing datasets.
        4. Applies preprocessing:
            - Fits and transforms the training input features.
            - Transforms the testing input features using the fitted preprocessor.
        5. Combines transformed features with target columns into final NumPy arrays.
        6. Saves the preprocessor object (pipeline) as a `.pkl` file for future inference use.
        """

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Reading training and testing data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on train and test dataframes")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df) # to learn parameters and transform it simultaneously
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df) # to transform it using the parameters learned from the training data
        
            # Concatenates the transformed input features with the original target labels to prepare for model training.
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            # It stacks the input features and target variable side-by-side, resulting in a single 2D NumPy array.

            logging.info("Saving preprocessing object as a pickle file.")

            # Saving as a pickle file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
