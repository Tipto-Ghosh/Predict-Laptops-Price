import pandas as pd 
import numpy as np
import re
from sklearn.base import BaseEstimator , TransformerMixin
from laptopPrice.logger import logging
from laptopPrice.exception import LaptopException
from laptopPrice.constants import SCHEMA_FILE_PATH
from laptopPrice.utils.common_utils import read_yaml_file


class FeatureEngineer(BaseEstimator , TransformerMixin):
    def __init__(self , schema_file_path: str = SCHEMA_FILE_PATH):
        super().__init__()
        self._schema_config = read_yaml_file(file_path = schema_file_path)
        
        
    def fit(self , X , y = None):
        return self
    
    def transform(self , X: pd.DataFrame , y = None):
        X = X.copy()
        
        if y is not None:
            y = y.copy()
            
        # columns need to drop
        drop_cols_list = self._schema_config['drop_columns']
        # drop the cols 
        X.drop(columns = drop_cols_list , axis = 1 , inplace = True)
        logging.info(f"Dropped cols: {drop_cols_list}")
        
        X['Weight']= X['Weight'].str.replace("kg" , "")
        X['Weight'] = X['Weight'].astype(float) 
        X['Ram'] = X['Ram'].str.replace("GB" , "")
        X['Ram'] = X['Ram'].astype(int)
        
        # Extract resolution using regex
        X['resX'] = X['ScreenResolution'].str.extract(r'(\d{3,4})x\d{3,4}').astype(int)
        X['resY'] = X['ScreenResolution'].str.extract(r'\d{3,4}x(\d{3,4})').astype(int)
        
        # Using Inches , resX and resY make a single feature PPI(Pixel Per Inch)
        X['ppi'] = np.sqrt(X['resX']**2 + X['resY']**2) / X['Inches']
        
        # drop resX , resY
        X.drop(columns = ["resX" , "resY" , "Inches"] , axis = 1 , inplace = True)
        
        # make a new feature called is_ips
        X['is_ips'] = X['ScreenResolution'].str.contains('IPS' , case = False , na = False).astype(int)
        
        X[X['ScreenResolution'].str.contains('Touchscreen' , case = False , na = False)]
        
        # make a new feature called is_touchscreen
        X['is_touchscreen'] = X['ScreenResolution'].str.contains('Touchscreen' , case = False , na = False).astype(int)
        
        # make 5 features: Intel Core i7 , Intel Core i5 , Intel Core i3 , Other Intel Processor , AMD Processor
        X['Cpu_name'] = X['Cpu'].apply(lambda x: ' '.join(x.split()[0 : 3]))
        
        X['Cpu_name'] = X['Cpu_name'].apply(self.fetch_cpu)
        
        # make clock speed(GHz)
        X['CPU_Speed_GHz'] = X['Cpu'].str.split(" ").str[-1].str.replace("GHz" , "").astype(float)
        # now drop both ScreenResolution and Cpu
        X.drop(columns = ['ScreenResolution' , 'Cpu'] , axis = 1 , inplace = True)
    
        
        # Apply to dataframe
        X['SSD_GB'] = X['Memory'].apply(lambda x: self.extract_storage(x, 'SSD'))
        X['HDD_GB'] = X['Memory'].apply(lambda x: self.extract_storage(x, 'HDD'))
        X['Flash_GB'] = X['Memory'].apply(lambda x: self.extract_storage(x, 'FLASH'))
        X['Hybrid_GB'] = X['Memory'].apply(lambda x: self.extract_storage(x, 'HYBRID'))
        
        X.drop(columns = ['Memory'] , axis = 1 , inplace = True)
        
        # as flash and hybrid dont have any high correlation so drop them 
        X.drop(columns = ['Flash_GB' , 'Hybrid_GB'] , axis = 1 , inplace = True)

        # extract gpu brand name 
        X['gpu_brand'] = X['Gpu'].str.split().str[0]
        # drop the gpu column
        X.drop(columns = ['Gpu'] , axis = 1 , inplace = True)
        X['OpSys'] = X['OpSys'].apply(self.cat_os)
        
        return X 
            
        
        
    def fetch_cpu(self , text): 
            if text == "Intel Core i5" or text == "Intel Core i7" or text == "Intel Core i3": 
                return text
            if text.split()[0] == "Intel": 
                return "other intel"
            else:
                return 'amd'
                
    
    def extract_storage(self , mem_str, storage_type):
        mem_str = str(mem_str).upper()
        size_gb = 0
        
        # Patterns for each type
        if storage_type == 'FLASH':
            pattern = r'(\d+\.?\d*)\s*GB\s*FLASH STORAGE'
        elif storage_type == 'HYBRID':
            pattern = r'(\d+\.?\d*)\s*TB\s*HYBRID|(\d+\.?\d*)\s*GB\s*HYBRID'
        else:
            pattern = rf'(\d+\.?\d*)\s*TB\s*{storage_type}|(\d+\.?\d*)\s*GB\s*{storage_type}'
        
        for match in re.finditer(pattern, mem_str):
            tb, gb = match.groups() if len(match.groups()) == 2 else (None, match.group(1))
            if tb:
                size_gb += float(tb) * 1024
            elif gb:
                size_gb += float(gb)
                
        return size_gb    
    
    # make os: windows , mac , Linux , other 
    def cat_os(self , text: str): 
        text = text.lower()
        
        if 'windows' in text: 
            return 'windows' 
        elif 'linux' in text: 
            return 'linux' 
        elif 'mac' in text: 
            return 'mac' 
        else:
            return 'other'
        
