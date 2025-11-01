import os
import sys 
from typing import Optional
import numpy as np
from sklearn.metrics import r2_score

from laptopPrice.exception import LaptopException
from laptopPrice.logger import logging
from laptopPrice.constants import TARGET_COLUMN
from laptopPrice.entity.estimator import LaptopPriceEstimator
from laptopPrice.entity.config_entity import ModelEvaluationConfig
from laptopPrice.entity.artifact_entity import  ModelEvaluationArtifact , ModelTrainerArtifact
from laptopPrice.utils.common_utils import load_object , read_csv


class ModelEvaluation:
    def __init__(self , model_evaluation_config: ModelEvaluationConfig , model_trainer_artifact: ModelTrainerArtifact , test_file_path: str):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.test_file_path = test_file_path
            
        except Exception as e:
            raise LaptopException(e , sys)
    
    
    def get_production_model(self) -> Optional[LaptopPriceEstimator]:
        """This function is used to get model in production

        Returns:
            Optional[WineQualityEstimator]: Returns model object if available in production
        """
        
        try:
            production_model_path = self.model_evaluation_config.production_model_path
            # check do we have any model?
            if not os.path.exists(production_model_path):
                logging.info("No production model file found at path: %s", production_model_path)
                return None 
            # load the model
            production_model_object = load_object(production_model_path)
            if production_model_object is None:
                logging.info("No model found in production")
                return None 
            else:
                logging.info(f"Model found in production. Type: {type(production_model_object)}")
                return production_model_object
        except Exception as e:
            raise LaptopException(e , sys)
    
    
    def evaluate_model(self) -> ModelEvaluationArtifact:
        """This function is used to evaluate trained model with production model and choose best model 

        Returns:
            ModelEvaluationArtifact: ModelEvaluationArtifact object
        """
        try:
            # load the transformed test data
            test_df = read_csv(self.test_file_path)
            logging.info(f"test dataframe loaded from evaluate_model. shape: [{test_df.shape}]")
            
            input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN] , axis = 1)
            target = np.log(test_df[TARGET_COLUMN])
            
            logging.info(f"from model evaluation seperated input and target feature. input shape:[{input_feature_test_df.shape}] , target_feature shape: [{target.shape}]")
            
            # load both new and production model
            new_trained_model = load_object(self.model_trainer_artifact.trained_estimator_object_file_path)
            
            if new_trained_model is None:
                logging.info("Failed to load new trained model")
                raise Exception("Failed to load new trained model")
            
            production_model = self.get_production_model()
            if production_model is not None: 
                y_pred_production_model = production_model.predict_dataframe(input_feature_test_df , acutal_price = False)
                production_model_score = r2_score(
                    y_true = target , y_pred = y_pred_production_model
                )
            else:
                production_model_score = 0.0
            
            # now calculate one new trained model
            y_pred_new_trained_model = new_trained_model.predict_dataframe(input_feature_test_df , acutal_price = False)
            new_trained_model_score = r2_score(
                y_true = target , y_pred = y_pred_new_trained_model
            )
            logging.info(f"New trained model r2_score on test data: {new_trained_model_score}")
            
            is_model_accepted = new_trained_model_score > production_model_score
            score_difference = new_trained_model_score - production_model_score
            
            if is_model_accepted:
                best_estimator_path = self.model_trainer_artifact.trained_estimator_object_file_path
            else:
                best_estimator_path = self.model_evaluation_config.production_model_path # estimator: feature egineer + transformer + model
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted = is_model_accepted,
                improved_score = score_difference,
                best_model_path = best_estimator_path                  
            )
            logging.info(f"result: {model_evaluation_artifact}")
            logging.info("Exiting from evaluate_model method of ModelEvaluation class")
            return model_evaluation_artifact
            
        except Exception as e:
            raise LaptopException(e , sys)
    
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """This method is used to initialize all the steps of the model evaluation

        Returns:
            ModelEvaluationArtifact: ModelEvaluationArtifact object
        """
        try:
            return self.evaluate_model()
        except Exception as e:
            raise LaptopException(e , sys)