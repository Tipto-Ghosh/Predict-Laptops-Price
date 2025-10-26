import os 
from dotenv import load_dotenv 

load_dotenv()


# database related constants
DATABASE_CONNECTION_URL = os.getenv("mongo_connection_url")
DATABASE_NAME = "laptop_price"
COLLECTION_NAME = "laptop_price_collection"

FILE_NAME = "laptop.csv" # Raw data file name

PIPELINE_NAME : str = "laptop_price"

ARTIFACT_DIR : str = "artifact"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
VALIDATION_FILE_NAME = "validation.csv"

PREPROCESSOR_OBJECT_FILE_NAME = "preprocessor.pkl"
MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN : str = "Price"
SCHEMA_FILE_PATH = os.path.join("config" , "schema.yaml")




# Data Ingestion related constants
DATA_INGESTION_COLLECTION_NAME : str = COLLECTION_NAME # mongo collection name
DATA_INGESTION_DIR_NAME : str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR : str = "feature_store"
DATA_INGESTION_INGESTED_DIR : str = "ingested"
DATA_INGESTION_TEST_SIZE : float = 0.15 # test set size
DATA_INGESTION_VALIDATION_SIZE : float = 0.15 # Validation set size