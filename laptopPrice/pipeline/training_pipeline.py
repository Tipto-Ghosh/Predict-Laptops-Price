import sys 
import warnings
warnings.filterwarnings("ignore")

from laptopPrice.logger import logging
from laptopPrice.exception import LaptopException

from laptopPrice.entity.config_entity import (
    DataIngestionConfig
)

from laptopPrice.entity.artifact_entity import (
    DataIngestionArtifact
)

from laptopPrice.components.data_ingestion import DataIngestion



class TrainingPipeline:
    def __init__(self):
        """This class is responsible to run the full training pipeline.
        """
        # 1. do the data ingestion
        self.data_ingestion_config = DataIngestionConfig()
    
    
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """ 
        This will start the data ingestion component and return DataIngestionArtifact
        """
        try:
            logging.info("Entered start_data_ingestion method from TrainingPipeline class")
            
            data_ingestion = DataIngestion(
                data_ingestion_config = self.data_ingestion_config
            ) 
            
            logging.info("Calling initiate data ingestion from start_data_ingestion method of TrainingPipeline class")
            
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion() 
            logging.info(f"data_ingestion_artifact is received from start_data_ingestion method. {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise LaptopException(e , sys)
    
    
    def run_training_pipeline(self):
        """ 
        This method of TrainingPipeline class is responsible for running complete training pipeline
        """
        
        try:
            # 1. Run the data ingestion  
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data Ingestion Completed")
        except Exception as e:
            raise LaptopException(e , sys)