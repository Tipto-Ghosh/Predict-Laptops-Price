from laptopPrice.entity.config_entity import DataValidationConfig
from laptopPrice.utils.common_utils import read_yaml_file , load_object
import pandas as pd 
from pathlib import Path

model = load_object("E:/end-to-end-machine-learning-project/Predict-Laptops-Price/artifacts/11_01_2025_17_39_04/model_trainer/trained_model/estimator.pkl")

validation_df = pd.read_csv("E:/end-to-end-machine-learning-project/Predict-Laptops-Price/artifacts/11_01_2025_17_39_04/data_ingestion/ingested/validation.csv")
test_df = pd.read_csv("E:/end-to-end-machine-learning-project/Predict-Laptops-Price/artifacts/11_01_2025_17_39_04/data_ingestion/ingested/test.csv")



X = test_df.drop(columns = ["Price"] , axis = 1)
y = test_df['Price']

X1 = validation_df.drop(columns = ["Price"] , axis = 1)
y1 = validation_df['Price']

pred = model.predict_dataframe(X)
pred1 = model.predict_dataframe(X1)


from sklearn.metrics import r2_score

print(f"Validation Data r2_score: {r2_score(y1 , pred1)}")
print(f"Test Data r2_score: {r2_score(y , pred)}")