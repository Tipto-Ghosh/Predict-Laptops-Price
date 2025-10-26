import os 
import sys
import pymongo

from laptopPrice.logger import logging
from laptopPrice.exception import LaptopException
from laptopPrice.constants import DATABASE_NAME , DATABASE_CONNECTION_URL


# class to return a mongoDB connection to get access to the database
class MongoDbClient:
    client = None 
    
    def __init__(self , mongo_db_url : str = DATABASE_CONNECTION_URL , database_name: str = DATABASE_NAME):
        try:
            if MongoDbClient.client is None: # no client available so make the connection
                # check connection string is available or not
                if mongo_db_url is None:
                   logging.info("Database Connection is not available in constants")
                   raise LaptopException("Database connection string is missing" , sys)
                
                # check database name is available or not
                if database_name is None:
                   logging.info("Database Name is not available in constants")
                   raise LaptopException("Database name is missing" , sys)
                
                # now connection string and database name is available so make the connection
                MongoDbClient.client = pymongo.MongoClient(mongo_db_url)
                self.client = MongoDbClient.client # also making a object level client(optional)
                self.database_name = database_name
                self.database = self.client.get_database(self.database_name)
                
                logging.info("MongoDB Connection Successfull") 
        except Exception as e:
            raise LaptopException(e , sys)