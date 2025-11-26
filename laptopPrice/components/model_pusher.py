import os 
import sys 
import shutil

from laptopPrice.exception import LaptopException
from laptopPrice.logger import logging

from laptopPrice.entity.artifact_entity import ModelEvaluationArtifact , ModelPusherArtifact
from laptopPrice.entity.config_entity import ModelPusherConfig


class ModelPusher:
    def __init__(self , model_pusher_config: ModelPusherConfig , model_evaluation_artifact: ModelEvaluationArtifact) -> None:
        
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise LaptopException(e , sys)
    
    
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Pushes the best model + feature engineer + preprocessor to production.
        If they exist, replaces them. If not, creates them.
        """
        try:
            # if model is not accepted
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted. Skipping push to production.")
                return ModelPusherArtifact(
                   is_model_pushed = False,
                   production_model_path = self.model_pusher_config.production_model_path
                )
            
            logging.info(f"Start Model Pushing to the production")
            
            # make production directory(For the first run there will be no directory)
            os.makedirs(os.path.dirname(self.model_pusher_config.production_model_path) , exist_ok = True)
            
            # replace the model 
            shutil.copy(
                src = self.model_evaluation_artifact.best_model_path,
                dst = self.model_pusher_config.production_model_path
            )
            logging.info(f"Model pushed to production: {self.model_pusher_config.production_model_path}")
            
            return ModelPusherArtifact(
                is_model_pushed = True,
                production_model_path = self.model_pusher_config.production_model_path
            )
              
        except Exception as e:
             raise LaptopException(e , sys)
