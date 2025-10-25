# Example usage
import sys
from laptopPrice.exception import LaptopException


try:
    a = 12 / 0
except ZeroDivisionError as e:
    raise LaptopException("Zero division error", sys)
