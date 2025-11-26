import os 
from dotenv import load_dotenv 

load_dotenv()


# database related constants
DATABASE_CONNECTION_URL = os.getenv("mongo_connection_url")
DATABASE_NAME = "laptop_price"
COLLECTION_NAME = "laptop_price_collection"

FILE_NAME = "laptop.csv" # Raw data file name

PIPELINE_NAME : str = "laptop_price"

ARTIFACT_DIR : str = "artifacts"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
VALIDATION_FILE_NAME = "validation.csv"

PREPROCESSOR_OBJECT_FILE_NAME = "preprocessor.pkl"
FEATURE_ENGINEERING_FILE_NAME = "feature_engineering_object.pkl"
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


# Data validation Constants
DATA_VALIDATION_DIR_NAME : str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR : str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME : str = "report.yaml"


# Data Transformation Constants
DATA_TRANSFORMATION_DIR_NAME : str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR : str = "transformed_data"
DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_DIR : str = "transformed_object"


# Model Trainer realted contant start with MODEL_TRAINER
MODEL_TRAINER_DIR_NAME : str = "model_trainer"
# trained model path
MODEL_TRAINER_TRAINED_MODEL_DIR : str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME : str = "recent_trained_model.pkl"
# Path to model.yaml
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH : str = os.path.join("config" , "params.yaml")
MODEL_TRAINER_EXPECTED_SCORE : float = 0.75 # minimal r2 score
# Reports directory for tuned models
MODEL_TRAINER_ALL_MODEL_REPORT_DIR: str = "all_model_report"
# File path where all tuned models' details will be saved
MODEL_TRAINER_ALL_TUNED_MODEL_REPORT_FILE_PATH: str = "all_tuned_model_report.yaml"
MODEL_TRAINER_ESTIMATOR_OBJECT_FILE_NAME : str = "estimator.pkl"


# Model Evaluation related constants
PRODUCTION_MODEL_PATH : str = os.path.join("Model" , "estimator.pkl")