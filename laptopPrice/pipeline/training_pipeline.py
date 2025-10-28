import sys 
import warnings
warnings.filterwarnings("ignore")

from laptopPrice.logger import logging
from laptopPrice.exception import LaptopException

from laptopPrice.entity.config_entity import (
    DataIngestionConfig , DataValidationConfig
)

from laptopPrice.entity.artifact_entity import (
    DataIngestionArtifact , DataValidationArtifact
)

from laptopPrice.components.data_ingestion import DataIngestion
from laptopPrice.components.data_validation import DataValidation



class TrainingPipeline:
    def __init__(self):
        """This class is responsible to run the full training pipeline.
        """
        # 1. do the data ingestion
        self.data_ingestion_config = DataIngestionConfig()
        # 2. do the data validation
        self.data_validation_config = DataValidationConfig()
    
    
    
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
    
    def start_data_validation(self , data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """This will start the data validation component. 
        Returns:
            DataValidationArtifact: 
        """
        
        try:
            logging.info("Entered start_data_validation method from TrainingPipeline class")
            
            data_validation = DataValidation(
                data_ingestion_artifact = data_ingestion_artifact ,
                data_validation_config = self.data_validation_config
            )
            
            logging.info("Calling initiate data validation from start_data_validation method of TrainingPipeline class")
            
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Performed the data validation operation")
            
            return data_validation_artifact
        
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
            
            # 2. Run the data validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact = data_ingestion_artifact
            )
            logging.info("Data Validation Completed")
        except Exception as e:
            raise LaptopException(e , sys)