import sys 
import os 
import pandas as pd 
import numpy as np 

from laptopPrice.logger import logging
from laptopPrice.exception import LaptopException
from laptopPrice.configuration.mongo_connection import MongoDbClient


class LaptopData:
    """ 
    This class helps to export entire mongo db record as pandas dataframe
    """
    
    def __init__(self):
        # get the database connection
        try:
            self.mongo_client = MongoDbClient() 
        except Exception as e:
            raise LaptopException(e , sys)
    
    def export_collection_data_as_dataframe(self , collection_name: str) -> pd.DataFrame:
        """Get all the data from the database collection as dict.
        Convert the dict into dataframe.
        Drop the mongodb default _id value.

        Args:
            collection_name (str): from which collection we want to export the data.

        Returns:
            pd.DataFrame: dataframe object.
        """
        
        try:
            collection = self.mongo_client.client[collection_name]
            
            # get all the data and convert them into dataframe
            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Got the dataframe from mongoDb. Shape: [{df.shape}]")
            
            # remove the _id column
            if '_id' in df.columns.to_list():
                df.drop(columns = ['_id'] , axis = 1 , inplace = True) 
            
            logging.info(f"After dropping _id column dataframe shape: [{df.shape}]")
            
            # replace na with NaN
            df.replace({"na" : np.nan} , inplace = True)
            
            return df 
        except Exception as e:
            raise LaptopException(e , sys)