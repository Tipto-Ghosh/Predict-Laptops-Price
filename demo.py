# Example usage
import sys
from laptopPrice.exception import LaptopException
from laptopPrice.data_access import LaptopData
from laptopPrice.constants import *
lp = LaptopData()
df = lp.export_collection_data_as_dataframe(COLLECTION_NAME)
print(df.shape)