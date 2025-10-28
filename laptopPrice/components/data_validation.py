import os
import sys 
import json 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from pandas import DataFrame
import pandera as pa
from pandera import Column, Check, DataFrameSchema



from laptopPrice.logger import logging
from laptopPrice.exception import LaptopException
from laptopPrice.entity.config_entity import DataValidationConfig
from laptopPrice.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact
from laptopPrice.utils.common_utils import read_yaml_file , read_csv

class DataValidation:
    def __init__(self , data_ingestion_artifact: DataIngestionArtifact , data_validation_config: DataValidationConfig):
        """This class is responsible to do data validation.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): object of DataIngestionArtifact
            data_validation_config (DataValidationConfig, optional): _description_. Defaults to DataValidationConfig().
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            logging.info("reading the schema file from DataValidation")
            self._schema_config = read_yaml_file(self.data_validation_config.schema_file_path)
            logging.info("reading the schema file from DataValidation is done")
        except Exception as e:
           raise LaptopException(e , sys) 
    
    def validate_numerical_columns(self , dataframe: pd.DataFrame) -> bool:
        """Validate all the numerical feature exists or not.

        Args:
            dataframe (DataFrame): pass dataframe for which we want to check

        Returns:
            bool: if all numerical columns exists then return True otherwise False
        """
        # get all the columns from the dataframe
        logging.info(f"DataFrame dtypes:\n{dataframe.dtypes}")
        
        dataframe_numerical_columns = dataframe.select_dtypes(include=['number']).columns.to_list()
        numerical_columns = self._schema_config['numerical_columns']
        
        logging.info(f"Loaded Numerical columns from schema: {numerical_columns}")
        
        missing_num_cols = []
        
        # if schema's columns are missing
        for col in numerical_columns:
            if col not in dataframe_numerical_columns:
                missing_num_cols.append(col)
        
        extra = []
        # if dataframe has extra numerical col
        for col in dataframe_numerical_columns:
            if col not in numerical_columns:
                extra.append(col) 
        
        if len(missing_num_cols) > 0:
            logging.info(f"Dataframe has missing numerical col: {missing_num_cols}")
            return False
        
        if len(extra) > 0:
            logging.info(f"Dataframe has extra numerical col: {extra}")
            return False
        
        return True
    
    
    def validate_categorical_columns(self , dataframe: pd.DataFrame) -> bool:
        """Validate all the categorical feature exists or not.

        Args:
            dataframe (DataFrame): pass dataframe for which we want to check

        Returns:
            bool: if all categorical columns exists then return True otherwise False
        """
        # get all the columns from the dataframe
        dataframe_categorical_columns = dataframe.select_dtypes(exclude = ['number']).columns.to_list()
        categorical_columns = self._schema_config['categorical_columns']
        
        logging.info(f"Loaded Categorical columns from schema: {categorical_columns}")
        
        missing_num_cols = []
        
        # if schema's columns are missing
        for col in categorical_columns:
            if col not in dataframe_categorical_columns:
                missing_num_cols.append(col)
        
        extra = []
        # if dataframe has extra categorical col
        for col in dataframe_categorical_columns:
            if col not in categorical_columns:
                extra.append(col) 
        
        if len(missing_num_cols) > 0:
            logging.info(f"Dataframe has missing categorical col: {missing_num_cols}")
            return False
        
        if len(extra) > 0:
            logging.info(f"Dataframe has extra categorical col: {extra}")
            return False
        
        return True
    
    
    def validate_data_using_pandera(self , dataframe: pd.DataFrame) ->None:
        """
        Validate a pandas DataFrame using a schema defined in a YAML file.

        Args:
            dataframe (pd.DataFrame): DataFrame to validate
            schema_yaml_path (str): Path to the YAML file containing schema definition

        Raises:
            LaptopException: If validation fails or schema file is invalid
        """
        
        try:
            logging.info("Starting dataframe validation using pandera schema...")
           
            # get the schema section
            pandera_columns = self._schema_config['pandera_columns']
           
            columns_schema = {}
           
            for col , props in pandera_columns.items():
                dtype = str if props.get("dtype") == str else float
                checks = []
            
                # Allowed values check
                if "allowed_values" in props:
                    checks.append(Check.isin(props["allowed_values"]))
                
                # Range check
                if "range" in props:
                    range_data = props["range"]
                    checks.append(Check.in_range(range_data["min"] , range_data["max"]))
                
                # create column object
                columns_schema[col] = Column(dtype , checks = checks , nullable = props.get("nullable", False)) 
            
            # Create full Pandera schema
            schema = DataFrameSchema(columns_schema)
            # Validate dataframe
            schema.validate(dataframe)
            logging.info("Data validation passed using pandera.")
            return True 
        except pa.errors.SchemaError as e:
            logging.info("Data validation failed using pandera.")
            return False
        
        except Exception as e:
            logging.error(f"Error occurred while performing Pandera validation")
            raise LaptopException(e , sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """This method is responsible to do the whole data validation part and make the data validation artifact.

        Returns:
            DataValidationArtifact: Object of DataValidationArtifact
        """
        try:
            logging.info("Starting data validation")
            train_df = read_csv(
                file_path = self.data_ingestion_artifact.train_file_path
            )
            test_df = read_csv(
                file_path = self.data_ingestion_artifact.test_file_path
            )
            validation_df = read_csv(
                file_path = self.data_ingestion_artifact.validation_file_path
            )
            logging.info(f"Loaded train set: shape[{train_df.shape}]")
            logging.info(f"Loaded test set: shape[{test_df.shape}]")
            logging.info(f"Loaded validation set: shape[{validation_df.shape}]")
            
            if self.validate_numerical_columns(train_df) == False:
                raise Exception("Missing numerical column in train_df")
            if self.validate_numerical_columns(test_df) == False:
                raise Exception("Missing numerical column in test_df")
            if self.validate_numerical_columns(validation_df) == False:
                raise Exception("Missing numerical column in validation_df")
            
            if self.validate_categorical_columns(train_df) == False:
                raise Exception("Missing categorical column in train_df")
            if self.validate_categorical_columns(test_df) == False:
                raise Exception("Missing categorical column in test_df")
            if self.validate_categorical_columns(validation_df) == False:
                raise Exception("Missing categorical column in validation_df")
            
            # if self.validate_data_using_pandera(train_df) == False:
            #     raise Exception("train df pandera test failed")
            # if self.validate_data_using_pandera(test_df) == False:
            #     raise Exception("test df pandera test failed")
            # if self.validate_data_using_pandera(validation_df) == False:
            #     raise Exception("validation df pandera test failed")
            
            data_validation_artifact = DataValidationArtifact(
                train_file_path = self.data_ingestion_artifact.train_file_path,
                test_file_path = self.data_ingestion_artifact.test_file_path,
                validation_file_path = self.data_ingestion_artifact.validation_file_path,
                data_validation_status = True
            )
            logging.info(f"Data validation completed and artifact created.")
            return data_validation_artifact
        except Exception as e:
            raise LaptopException(e , sys)
