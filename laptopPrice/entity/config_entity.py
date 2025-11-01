import os 
from laptopPrice.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP : str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name : str = PIPELINE_NAME
    artifact_dir : str = os.path.join(ARTIFACT_DIR , TIMESTAMP)
    timestamp = TIMESTAMP

training_pipeline_config : TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir : str = os.path.join(training_pipeline_config.artifact_dir , DATA_INGESTION_DIR_NAME)
    feature_store_file_path : str = os.path.join(data_ingestion_dir , DATA_INGESTION_FEATURE_STORE_DIR , FILE_NAME)
    training_file_path : str = os.path.join(data_ingestion_dir , DATA_INGESTION_INGESTED_DIR , TRAIN_FILE_NAME)
    testing_file_path : str = os.path.join(data_ingestion_dir , DATA_INGESTION_INGESTED_DIR , TEST_FILE_NAME)
    validation_file_path : str = os.path.join(data_ingestion_dir , DATA_INGESTION_INGESTED_DIR , VALIDATION_FILE_NAME)
    test_size : float = DATA_INGESTION_TEST_SIZE
    validation_size : float = DATA_INGESTION_VALIDATION_SIZE
    collection_name : str = DATA_INGESTION_COLLECTION_NAME
    

@dataclass
class DataValidationConfig:
    # artifact/timestamp/data_validation/
    data_validation_dir : str = os.path.join(training_pipeline_config.artifact_dir , DATA_VALIDATION_DIR_NAME)
    # artifact/timestamp/data_validation/drift_report/report.yaml
    data_drift_file_path : str = os.path.join(data_validation_dir , DATA_VALIDATION_DRIFT_REPORT_DIR , DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
    # schema file path
    schema_file_path : str = SCHEMA_FILE_PATH


@dataclass
class DataTransformationConfig:
    # artifact/timestamp/data_transformation/
    data_transformation_dir : str = os.path.join(training_pipeline_config.artifact_dir , DATA_TRANSFORMATION_DIR_NAME)
    # artifact/timestamp/data_transformation/transformed_data/train.npy
    transformed_train_data_file_path : str = os.path.join(
        data_transformation_dir , DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR , 
        TRAIN_FILE_NAME.replace("csv" , "npy")
    )
    # artifact/timestamp/data_transformation/transformed_data/validation.npy
    transformed_validation_data_file_path : str = os.path.join(
        data_transformation_dir , DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR , 
        VALIDATION_FILE_NAME.replace("csv" , "npy")
    )
    # artifact/timestamp/data_transformation/transformed_object/preprocessor.pkl
    transformed_object_file_path : str = os.path.join(
        data_transformation_dir , DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_DIR , PREPROCESSOR_OBJECT_FILE_NAME 
    )
    # artifact/timestamp/data_transformation/transformed_object/feature_engineering_object.pkl
    feature_engineering_object_file_path : str = os.path.join(
        data_transformation_dir , DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_DIR , FEATURE_ENGINEERING_FILE_NAME 
    )


@dataclass
class ModelTrainerConfig:
    # artifact/timestamp/model_trainer
    model_trainer_dir : str = os.path.join(training_pipeline_config.artifact_dir , MODEL_TRAINER_DIR_NAME)
    # artifact/timestamp/model_trainer/trained_model/model.pkl
    trained_model_file_path : str = os.path.join(
        model_trainer_dir , MODEL_TRAINER_TRAINED_MODEL_DIR , MODEL_TRAINER_TRAINED_MODEL_NAME
    )
    expected_accuracy : float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path : str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
    
    # directory to store all tuned model reports
    # artifact/timestamp/model_trainer/"all_model_report"/"all_tuned_model_report.yaml" 
    all_models_report_file_path : str = os.path.join(
        model_trainer_dir , MODEL_TRAINER_ALL_MODEL_REPORT_DIR , MODEL_TRAINER_ALL_TUNED_MODEL_REPORT_FILE_PATH
    )
    trained_estimator_object_file_path : str = os.path.join(
        model_trainer_dir , MODEL_TRAINER_TRAINED_MODEL_DIR , MODEL_TRAINER_ESTIMATOR_OBJECT_FILE_NAME
    )