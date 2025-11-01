import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error

from laptopPrice.logger import logging
from laptopPrice.exception import LaptopException
from laptopPrice.utils.model_factory import ModelFactory

from laptopPrice.entity.config_entity import DataTransformationConfig , ModelTrainerConfig
from laptopPrice.entity.artifact_entity import ModelTrainerArtifact , DataTransformationArtifact , RegressionMetricArtifact
from laptopPrice.entity.estimator import LaptopPriceEstimator
from laptopPrice.utils.common_utils import load_numpy_array_data , load_object , save_object


class ModelTrainer:
    def __init__(self , model_trainer_config: ModelTrainerConfig , data_transformation_artifact: DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact
    
    def get_model_object_and_report(self , train: np.array , test: np.array) -> Tuple[object , object]:
        """ 
        Description :   This function uses ModelFactory to get the best model object and report of the best model.
        Returns metric artifact object and best model object
        """
        try:
            logging.info("Entered into get_model_object_and_report method of ModelTrainer class")
            
            # 1. seperate input feature and target from train data
            logging.info("1. seperate input feature and target from train data")
            X_train , y_train = train[ : , : -1] , train[ : , -1]
            
             # 2. seperate input feature and target from test data
            logging.info("2. seperate input feature and target from test data")
            X_test , y_test = test[ : , : -1] , test[ : , -1]
            
            # 3. use ModelFactory to get the best model object
            model_factory = ModelFactory(
                model_config_path = self.model_trainer_config.model_config_file_path,
                tuned_model_report_path = self.model_trainer_config.all_models_report_file_path
            )
            
            # run the model factory to do the hyper-parameter tuning
            model_factory.run_model_factory(
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test
            )
            
            # tune all the models from model.yaml file
            logging.info("calling model_factory.get_best_model method from get_model_object_and_report")
            best_model_detail = model_factory.get_best_model()
            
            # get the best model
            module_name = best_model_detail.module_name
            class_name = best_model_detail.model_name 
            best_params = best_model_detail.best_params
            logging.info(f"got the best model from model trainer[model name = {class_name}]")
            logging.info(f"best model type [{type(best_model_detail.best_model)}]")
            
            # get the best model object
            model_obj = best_model_detail.best_model
            logging.info("Got best model from best_model_detail.best_model")
            
            logging.info("Started train the best model object")
            model_obj.fit(X_train , y_train)
            
            # 4. do the prediction using on test data with the best model
            logging.info("started the prediction using on test data with the best model")
            y_pred = model_obj.predict(X_test)
            logging.info(f"prediction done with best model object. y_pred shape: ({y_pred.shape})")
            
            # 5. find the regression metrices for test data
            logging.info("finding the regression metrices for test data") 
            r2 = r2_score(y_test , y_pred)
            mae = mean_absolute_error(y_test , y_pred)
            mse = mean_squared_error(y_test , y_pred)
            
            
            # 6. make the RegressionMetricArtifact
            logging.info("making the RegressionMetricArtifact")
            metric_artifact = RegressionMetricArtifact(
                mean_absolute_error = mae,
                r2_score = r2,
                mean_squared_error = mse
            )
            
            # 7. return the best_model details and RegressionMetricArtifact
            logging.info("Exiting from get_model_object_and_report method")
            return best_model_detail , metric_artifact
        
        except Exception as e:
            raise LaptopException(e , sys)
    
    
    
    def initiate_model_trainer(self , ) -> ModelTrainerArtifact:
        """ 
        This function initiates a model trainer steps for training pipeline
        """
        
        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")
            
            # 1. load the train and test numpy array
            train_arr = load_numpy_array_data(
                file_path = self.data_transformation_artifact.transformed_train_data_file_path
            )
            validation_arr = load_numpy_array_data(
                file_path = self.data_transformation_artifact.transformed_validation_data_file_path
            )
            
            # 2. call  get_model_object_and_report to get the best model
            best_model_detail , metric_artifact = self.get_model_object_and_report(
                train = train_arr , test = validation_arr
            )
            
            # 3. check best model accepted or not based on expected_score
            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than expected score")
                raise Exception("No best model found with score more than expected score")
            
            # 4. Call UsVisaModel to combine preproccessor and best model
            # load the preprocessing object
            logging.info("load the preprocessing object for merging with best model")
            preprocessing_object = load_object(self.data_transformation_artifact.transformed_object_file_path)
            
            logging.info("load the feature engineering object")
            feature_engineer_object = load_object(self.data_transformation_artifact.feature_engineering_object_file_path)
            laptopPriceEstimator = LaptopPriceEstimator(
                feature_engineering_object = feature_engineer_object,
                preprocessing_object = preprocessing_object , 
                trained_model_object = best_model_detail.best_model
            )
            logging.info("saving LaptopPriceEstimator(feature_object + preprocessing_object + best_model_detail.best_model)")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path , obj = best_model_detail.best_model
            )
            
            # 6. construct the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                metric_artifact = metric_artifact,
                tuned_model_report_file_path = self.model_trainer_config.all_models_report_file_path
            )
            # 7. return the model trainer artifact
            return (model_trainer_artifact , laptopPriceEstimator)
        except Exception as e:
            raise LaptopException(e , sys)