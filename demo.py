from laptopPrice.entity.config_entity import DataValidationConfig
from laptopPrice.utils.common_utils import read_yaml_file


# dvc = DataValidationConfig()
# schema_config = read_yaml_file(dvc.schema_file_path)

# print(schema_config.keys())

# print(schema_config['numerical_columns'])

# print(schema_config['pandera_columns'].keys())

import pandas as pd 
from pathlib import Path  

train_df_path = Path("artifacts/10_28_2025_15_40_40/data_ingestion/ingested/train.csv")
# print(train_df_path)
train_df = pd.read_csv(train_df_path)

print(train_df.head())
print(train_df.columns)