from laptopPrice.entity.config_entity import DataValidationConfig
from laptopPrice.utils.common_utils import read_yaml_file


# dvc = DataValidationConfig()
# schema_config = read_yaml_file(dvc.schema_file_path)


from laptopPrice.feature_engineering.feature_engineer import FeatureEngineer 
import pandas as pd 

df = pd.read_csv("E:/end-to-end-machine-learning-project/Predict-Laptops-Price/notebooks/laptop_data.csv")

X = df.drop(columns = ["Price"] , axis = 1)
y = df['Price']

fe = FeatureEngineer()
fe.fit(X , y)
X = fe.transform(X , y)

print(X.head())