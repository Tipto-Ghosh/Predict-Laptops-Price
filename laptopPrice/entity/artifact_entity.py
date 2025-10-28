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
    