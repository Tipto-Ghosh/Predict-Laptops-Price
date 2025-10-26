import os 
import sys 

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from laptopPrice.logger import logging
from laptopPrice.exception import LaptopException
from laptopPrice.entity.config_entity import DataIngestionConfig
from laptopPrice.entity.artifact_entity import DataIngestionArtifact
from laptopPrice.data_access import LaptopData


class DataIngestion:
    pass  