import os
from pathlib import Path
import logging


logging.basicConfig(level = logging.INFO , format = "[%(asctime)s]: %(message)s:")

project_name = "laptopPrice"


list_of_files = [
    f"{project_name}/__init__.py",
    
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    
    f"{project_name}/configuration/__init__.py",
    
    f"{project_name}/constants/__init__.py",
    
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/common_utils.py",
    

    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    
    f"{project_name}/data_access/__init__.py",
    
    "config/params.yaml",
    "config/schema.yaml",
    
    "notebooks/mongoDbDataPush.ipynb",
    
    "main.py",
    "app.py",
    "demo.py",
    
    "requirements.txt",
    "setup.py",
    ".env",
    
    "static/css/style.css",
    "templates/index.html",
]


# make all the folders and files one by one
for filepath in list_of_files:
    filepath = Path(filepath)
    file_dir , file_name = os.path.split(filepath)
    
    # create directory if not exists
    if file_dir != "":
        os.makedirs(file_dir , exist_ok = True)
        logging.info(f"Creating directory: {file_dir}")
    
    # create file if not exists or empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filepath} already exists")