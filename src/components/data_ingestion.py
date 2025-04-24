"""
The Data Ingestion module is the first stage of your ML pipeline. It handles importing data from 
its source and organizing it into a form suitable for processing. This step ensures that the data 
pipeline starts with clean, structured, and reproducible access to raw and split datasets.

"""

import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass # Directily define class variable, makes the code cleaner
class DataIngestionConfig:  # For the inputs in data ingestion
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    """
    Data Ingestion component is responsible for ingesting the data from the source and saving it in 
    the specified format. \n
    Various steps includes: 
    1. Reading raw data
    2. Saving a copy of that raw data
    3. Splitting it into train and test sets
    4. Storing those split datasets into the artifacts/ folder

    This class ensures that all ingestion steps are logged and saved for 
    reproducibility and further processing.
    """

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self): 
        """
        This function is responsible for reading the raw data from the source and saving it in the specified format.\n
        """
       
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv('notebook/dataset/stud.csv')
            logging.info("Read the dataset as dataframe")
            
            # Creates the artifacts/ folder if it doesn’t exist.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            # Saves the raw data to artifacts/raw.csv.
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            
            logging.info("Train - Test split initiated")
            train_set, test_set = train_test_split(df, test_size = 0.3, random_state = 42)

            # Saves the split datasets into the artifacts/ folder.
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion is completed.")

            return( # Returns the paths so the next step in the pipeline (like data transformation) can use them.
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))


        