from laptopPrice.exception import LaptopException
from laptopPrice.constants import PRODUCTION_MODEL_PATH
from laptopPrice.utils.common_utils import load_object
from laptopPrice.logger import logging
import pandas as pd 
import sys 

class CustomData:
    def __init__(self , data_dict: dict):
        self.data_dict = data_dict
        
    def to_dataframe(self):
        # convert scalar values into single element lists
        row = {key : [value] for key , value in self.data_dict.items()}
        return pd.DataFrame(row) # return the dataframe
    

class PredictPipeline:
    def __init__(self):
        # store the path of the production model
        self.model = load_object(PRODUCTION_MODEL_PATH)
    
    def predict(self , custom_data: CustomData):
        try:
            logging.info("Prediction pipeline started")
            # convert the custom data into a dataframe
            df = custom_data.to_dataframe()
            # logging.info(f"---------------\n {df.info()} \n --------------")
            # Make prediction
            laptop_price = self.model.predict_user_info(df)[0]
            
            print(f"Predicted price: ${laptop_price:.2f}")
            
            # Return the prediction 
            return {
                'prediction': round(laptop_price, 2)
            } 
        except Exception as e:
            raise LaptopException(e , sys)