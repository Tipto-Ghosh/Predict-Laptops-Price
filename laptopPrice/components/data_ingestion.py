import os 
import sys 

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from laptopPrice.logger import logging
from laptopPrice.exception import LaptopException
from laptopPrice.entity.config_entity import DataIngestionConfig
from laptopPrice.entity.artifact_entity import DataIngestionArtifact
from laptopPrice.data_access import LaptopData
from laptopPrice.utils.common_utils import save_csv_file

class DataIngestion:
    def __init__(self , data_ingestion_config : DataIngestionConfig = DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config
    
    def export_data_into_feature_store(self) -> DataFrame:
        """ 
        This method exports data from the database and save as a csv file(raw data file)
        """
        try:
            logging.info("Entered into export_data_into_feature_store method.")
            
            laptop_data_obj = LaptopData()
            dataframe = laptop_data_obj.export_collection_data_as_dataframe(
                collection_name = self.data_ingestion_config.collection_name
            )
            
            logging.info("Data conversion from Collection to DataFrame successful")
            logging.info(f"Shape of the dataframe: {dataframe.shape}")
            
            # save the full raw dataset into feature store
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # feature_store_dir_name = os.path.dirname(feature_store_file_path)
            
            # make the directory
            # os.makedirs(feature_store_dir_name , exist_ok = True)
            
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            # dataframe.to_csv(feature_store_file_path , index = False , header = True)
            
            save_csv_file(
                file_path = feature_store_file_path , data = dataframe
            )
            
            return dataframe
             
        except Exception as e:
            raise LaptopException(e , sys)
    
    
    def split_data_as_train_test_validation_set(self , dataframe: DataFrame) -> None:
        """
        This method splits the dataframe into train set , test set and validation set based on split ratio.
        Then save the train , test and validation set as csv file inside ingested directory.
        """ 
        
        logging.info("Entered split_data_as_train_test method of data_ingestion")
        
        try:
            # step-1: split test set
            train_validation_data , test_set = train_test_split(
                dataframe , test_size = self.data_ingestion_config.test_size , random_state = 42
            )
            # Step 2: Calculate validation proportion relative to remaining train + val
            val_relative_size = self.data_ingestion_config.validation_size / (1 - self.data_ingestion_config.test_size)
            # step-3: split train and validation set
            train_set , validation_set = train_test_split(
                train_validation_data , test_size = val_relative_size , random_state = 42
            )
            
            logging.info("Data Splited into 3 parts. Train , Test and validation")
            logging.info(f"Train shape:[{train_set.shape}]. Test shape:[{test_set.shape}]. Validation shape:[{validation_set.shape}]")
            
            # now save these 3 data sets.
            save_csv_file(
                file_path = self.data_ingestion_config.training_file_path , data = train_set
            )
            save_csv_file(
                file_path = self.data_ingestion_config.testing_file_path , data = test_set
            )
            save_csv_file(
                file_path = self.data_ingestion_config.validation_file_path , data = validation_set
            )
            
            logging.info("Saved train , test and validation data as csv file")
        except Exception as e:
            raise LaptopException(e , sys)
    
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """This method initiates the data ingestion components of training pipeline.
           Merge to method export_data_into_feature_store and split_data_as_train_test_validation_set.
           
        Returns:
            DataIngestionArtifact: DataIngestionArtifact object 
        """
        
        try:
            # step-01: Get the full data
            dataframe = self.export_data_into_feature_store() 
            logging.info("from initiate_data_ingestion: Got the data from mongodb as Dataframe")
            
            # 2. do the train validation test set split
            self.split_data_as_train_test_validation_set(dataframe = dataframe)
            
            # 3. make the data ingestion artifact
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path = self.data_ingestion_config.training_file_path,
                test_file_path = self.data_ingestion_config.testing_file_path,
                validation_file_path = self.data_ingestion_config.validation_file_path
            )
            logging.info("Returning data_ingestion_artifact object from initiate_data_ingestion method")
            return data_ingestion_artifact

        except Exception as e:
            raise LaptopException(e , sys)
        