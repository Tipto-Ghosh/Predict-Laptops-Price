import sys 
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from laptopPrice.logger import logging
from laptopPrice.exception import LaptopException
from laptopPrice.constants import SCHEMA_FILE_PATH , TARGET_COLUMN
from laptopPrice.utils.common_utils import save_object , save_numpy_array_data , read_csv , read_yaml_file , drop_columns


class TargetValueMapping:
    def __init__(self):
        # mapping for actual log target value (can be extended)
        self.actual_price = {0: "Below 50K", 1: "50K-100K", 2: "Above 100K"}  
    
    def get_price(self, price: float) -> float:
        """Convert log to actual price"""
        return np.exp(price)


class LaptopPriceEstimator:
    def __init__(self , feature_engineering_object: object , preprocessing_object: Pipeline , trained_model_object: object):
        """This class is responsible to combine preprocessor and sklearn model.
           Also to do prediction for new data.

        Args:
            feature_engineering_object (object): _description_
            preprocessing_object (Pipeline): _description_
            trained_model_object (object): _description_
        """
        self.feature_engineering_object = feature_engineering_object
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self._schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)
        self.drop_cols = self._schema_config["drop_columns"]
    
    
    def predict_transformed_array(self , transformed_array : np.ndarray) -> np.ndarray:
        """
        Predict using already transformed feature array.
        """
        logging.info("Entered predict_array method of LaptopPriceEstimator class")
        
        try:
            predictions = self.trained_model_object.predict(transformed_array)
            logging.info(f"Prediction completed on transformed array, shape={predictions.shape}")
            return predictions 
        
        except Exception as e:
            raise LaptopException(e , sys)
    
    def predict_dataframe(self , input_df: DataFrame , acutal_price: bool = True) -> np.ndarray:
        """
        Transform raw input DataFrame and predict in one step.
        Expects input_df to have a single row or multiple rows with same feature columns.
        """
        logging.info(f"Entered predict_dataframe method with input shape: {input_df.shape}")
        try:
            df = input_df.copy()
            
            # Drop columns that were dropped in training
            # df = df.drop(columns = [col for col in self.drop_cols if col in df.columns] , errors = "ignore" , axis = 1)
            
            df = self.feature_engineering_object.transform(df)
            logging.info(f"df features after feature engineering: {df.columns}")
            # logging.info(f"Gpu: {df['Gpu']}")
            # Align columns with the training schema
            expected_features = getattr(self.preprocessing_object, "feature_names_in_", None)
            if expected_features is not None:
                missing_cols = [c for c in expected_features if c not in df.columns]
                logging.info(f"from LaptopPriceEstimator missing columns are:{missing_cols}")
                for c in missing_cols:
                    df[c] = 0
                df = df[expected_features]
            
            # transform using preprocessing_object
            transformed_data = self.preprocessing_object.transform(df)
            
            # predict
            predictions = self.trained_model_object.predict(transformed_data)
            
            # do the mapping
            if acutal_price:
                mapper = TargetValueMapping()
                predictions = [mapper.get_price(p) for p in predictions]
            
            return predictions
            
        except Exception as e:
            raise LaptopException(e , sys)
    
    def predict_user_info(self , input_df: DataFrame , acutal_price: bool = True) -> np.ndarray:
        """
        Transform raw input DataFrame and predict in one step.
        Expects input_df to have a single row or multiple rows with same feature columns.
        """
        logging.info(f"Entered predict_dataframe method with input shape: {input_df.shape}")
        try:
            # transform using preprocessing_object
            transformed_data = self.preprocessing_object.transform(input_df)
            
            # predict
            predictions = self.trained_model_object.predict(transformed_data)
            
            # do the mapping
            if acutal_price:
                mapper = TargetValueMapping()
                predictions = [mapper.get_price(p) for p in predictions]
            
            return predictions
        except Exception as e:
            raise LaptopException(e , sys)
    
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
