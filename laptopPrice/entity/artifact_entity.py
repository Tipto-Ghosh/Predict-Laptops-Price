from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path : str 
    test_file_path : str
    validation_file_path : str 
    

@dataclass
class DataValidationArtifact:
    train_file_path: str 
    test_file_path: str 
    validation_file_path: str 
    data_validation_status: bool 


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path : str
    transformed_train_data_file_path : str
    transformed_validation_data_file_path : str 
    feature_engineering_object_file_path : str


@dataclass
class RegressionMetricArtifact:
    r2_score : float
    mean_squared_error : float
    mean_absolute_error : float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path : str
    metric_artifact : RegressionMetricArtifact
    tuned_model_report_file_path : str