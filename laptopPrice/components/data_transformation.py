import sys 
import os 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd 
from sklearn.preprocessing import FunctionTransformer , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from laptopPrice.constants import SCHEMA_FILE_PATH , TARGET_COLUMN
from laptopPrice.logger import logging
from laptopPrice.exception import LaptopException

from laptopPrice.entity.config_entity import DataValidationConfig , DataTransformationConfig
from laptopPrice.entity.artifact_entity import DataValidationArtifact , DataTransformationArtifact

from laptopPrice.utils.common_utils import save_object , save_numpy_array_data , read_csv , read_yaml_file , drop_columns
from laptopPrice.feature_engineering.feature_engineer import FeatureEngineer
from laptopPrice.feature_engineering.mean_encoder import MeanEncoder


class DataTransformation:
    def __init__(self , data_transformation_config: DataTransformationConfig , data_validation_artifact: DataValidationArtifact):
        """This class is responsible for doing all the preprocessing task.

        Args:
            data_transformation_config (DataTransformationConfig): _description_
            data_validation_artifact (DataValidationArtifact): _description_
        """
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)
            
        except Exception as e:
            raise LaptopException(e , sys)
    
    def get_data_transformation_object(self) -> Pipeline:
        """ 
        This method creates and returns a data transformer(preprocessor) object for the data
        """
        try:
            logging.info("Entered get_data_transformer_object method of DataTransformation class") 
            preprocessor = Pipeline(steps = [
                ('mean_encoding', MeanEncoder()),
                ('scaling', StandardScaler())
            ])
            
            return preprocessor
            
        except Exception as e:
            raise LaptopException(e , sys) 
    
    def feature_engineering(self , X , y = None):
        """This method is responsible for doing all the feature engineering and save the object.

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
        """
        try:
            logging.info("Entered feature_engineering method of DataTransformation class") 
            feature_engineer = FeatureEngineer()
            
            X_transformed = feature_engineer.fit_transform(X , y)
            logging.info("After Feature engineering:")
            logging.info(X_transformed.head())
            
            save_object(
                file_path = self.data_transformation_config.feature_engineering_object_file_path,
                obj = feature_engineer
            )
            return X_transformed , feature_engineer
        
        except Exception as e:
            raise LaptopException(e , sys)
    

    def initiate_data_transformation(self , ) -> DataTransformationArtifact:
        """ 
        This method initiates the data transformation component for the pipeline
        """
        try:
            logging.info("Entered into initiate_data_transformation method")
            
            # check for data validation status
            if self.data_validation_artifact.data_validation_status: # if data validation status is true
                logging.info("Starting Data Transformation")
                
                # get the train and validation dataframe
                train_df = read_csv(file_path = self.data_validation_artifact.train_file_path)
                validation_df = read_csv(file_path = self.data_validation_artifact.validation_file_path)
                # dont load the test data, it should be in-take for model evaluation
                
                # separete target columns and input features[X , y]
                input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN] , axis = 1)
                target_feature_train_df = train_df[TARGET_COLUMN]  
                logging.info("Separeting X and y from train dataframe")
                
                # separete target columns and input features[X , y] from test data
                input_feature_validation_df = validation_df.drop(columns = [TARGET_COLUMN] , axis = 1)
                target_feature_validation_df = validation_df[TARGET_COLUMN]
                logging.info("Separeting X and y from validation dataframe")
                
                
                # do the feature engineering here
                # do the feature engineering
                input_feature_train_df , feature_engineer = self.feature_engineering(
                    X = input_feature_train_df , y = target_feature_train_df
                )
                
                input_feature_validation_df = feature_engineer.transform(input_feature_validation_df)
                logging.info("Applied feature engineering on train and validation data")
                
                
                
                # get the preprocessor object
                preprocessor = self.get_data_transformation_object()
                logging.info("Got the preprocessor object")
                
                # do the fit_transform on validation data
                logging.info("Applying preprocessing object on training dataframe and validation dataframe")
                
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df , target_feature_train_df)
                logging.info("Used the preprocessor object to fit transform the train features")
                
                # do the transformation on validation data
                input_feature_validation_arr = preprocessor.transform(input_feature_validation_df)
                logging.info("Used the preprocessor object to transform the validation features")
                
                # as y is normally distributed apply log transformation on it
                target_feature_train_df = np.log(target_feature_train_df)
                target_feature_validation_df = np.log(target_feature_validation_df)
                
                logging.info("Created train array and validation array")
                
                logging.info("concatenating input_feature_train_final arr and target_feature_train_final arr")
                train_arr = np.c_[
                    input_feature_train_arr , np.array(target_feature_train_df)
                ]
                
                logging.info("concatenating input_feature_validation_final arr and target_feature_validation_final arr")
                validation_arr = np.c_[
                    input_feature_validation_arr , np.array(target_feature_validation_df)
                ]
                
                # save the preprocessor object
                save_object(
                    file_path = self.data_transformation_config.transformed_object_file_path,
                    obj = preprocessor
                )
                logging.info("saved preprocessor object")
                
                # save the train and validation array's
                save_numpy_array_data(
                    file_path = self.data_transformation_config.transformed_train_data_file_path,
                    array = train_arr
                )
                logging.info("saved train arr")
                
                save_numpy_array_data(
                    file_path = self.data_transformation_config.transformed_validation_data_file_path,
                    array = validation_arr
                )
                logging.info("saved validation arr")
                
                # make the data transformation artifact
                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                    transformed_train_data_file_path = self.data_transformation_config.transformed_train_data_file_path,
                    transformed_validation_data_file_path = self.data_transformation_config.transformed_validation_data_file_path
                )
                logging.info("Exited initiate_data_transformation method of Data_Transformation class")
                return data_transformation_artifact
            else:
                logging.info("data transformation failed because of validation status")
                raise Exception("data transformation failed because of validation status")
        except Exception as e:
            raise LaptopException(e , sys)