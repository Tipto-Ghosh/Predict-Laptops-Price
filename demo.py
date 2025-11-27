from laptopPrice.entity.config_entity import DataValidationConfig
from laptopPrice.utils.common_utils import read_yaml_file , load_object
import pandas as pd 
import numpy as np
from pathlib import Path

# model = load_object("Model/estimator.pkl")
# expected_features = getattr(model.preprocessing_object, "feature_names_in_", None)

# print(expected_features)
# validation_df = pd.read_csv("E:/end-to-end-machine-learning-project/Predict-Laptops-Price/artifacts/11_01_2025_17_39_04/data_ingestion/ingested/validation.csv")
# test_df = pd.read_csv("E:/end-to-end-machine-learning-project/Predict-Laptops-Price/artifacts/11_01_2025_17_39_04/data_ingestion/ingested/test.csv")



# X = test_df.drop(columns = ["Price"] , axis = 1)
# y = test_df['Price']

# X1 = validation_df.drop(columns = ["Price"] , axis = 1)
# y1 = validation_df['Price']

# pred = model.predict_dataframe(X , acutal_price = True)
# pred1 = model.predict_dataframe(X1 , acutal_price = True)


# print(f"actual: {y} || pred: {pred}")

from sklearn.metrics import r2_score

# print(f"Validation Data r2_score: {r2_score(y1 , pred1)}")
# print(f"Test Data r2_score: {r2_score(y , pred)}")

# print(test_df.columns)


df = pd.read_csv("E:/end-to-end-machine-learning-project/Predict-Laptops-Price/artifacts/11_27_2025_16_06_20/data_ingestion/feature_store/laptop.csv")

from laptopPrice.feature_engineering.feature_engineer import FeatureEngineer

fe = FeatureEngineer()
df = fe.fit_transform(df)

# print(df.columns)

for col in df.columns.to_list():
    if col == 'Price':
        continue
    print(f"column name: {col}")
    print("Unique values are:")
    print(df[col].unique())
    print(" = " * 50)

inch = 15.6
x = 1920
y = 1080
# np.sqrt(X['resX']**2 + X['resY']**2) / X['Inches']
ppi = ((x**2 + y**2)**0.5)/inch

test_df = pd.DataFrame({
    'Company': ['HP'],
    'TypeName': ['Notebook'],
    'Ram': [4],
    'OpSys': ['No OS'],
    'Weight': [1.86],
    'ppi': [ppi],
    'is_ips': [0],
    'is_touchscreen': [0],
    'Cpu_name': ['Intel'],
    'CPU_Speed_GHz': [2],
    'SSD_GB': [0],
    'HDD_GB': [500],
    'gpu_brand': ['Intel']
})

# print(model.predict_user_info(test_df))