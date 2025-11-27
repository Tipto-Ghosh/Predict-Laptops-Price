from laptopPrice.exception import LaptopException
from laptopPrice.constants import PRODUCTION_MODEL_PATH
from laptopPrice.utils.common_utils import load_object
from flask import Flask, render_template, request, jsonify
import pandas as pd
import math

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    # Initial GET request: render index.html without prediction
    if request.method == "GET":
        return render_template('index.html')
    
    # POST request: Get the data and make prediction
    if request.method == "POST":
        try:
            # Get the JSON data from the request
            data = request.get_json()
            
            # Extract values from the request
            company = data.get('Company')
            type_name = data.get('TypeName')
            ram = data.get('Ram')
            op_sys = data.get('OpSys')
            weight = data.get('Weight')
            ppi = data.get('ppi')
            is_ips = data.get('is_ips')
            is_touchscreen = data.get('is_touchscreen')
            cpu_name = data.get('Cpu_name')
            cpu_speed = data.get('CPU_Speed_GHz')
            ssd_gb = data.get('SSD_GB')
            hdd_gb = data.get('HDD_GB')
            gpu_brand = data.get('gpu_brand')
            
            # Create the dataframe for prediction
            user_info_dataframe = pd.DataFrame({
                'Company': [company],
                'TypeName': [type_name],
                'Ram': [ram],
                'OpSys': [op_sys],
                'Weight': [weight],
                'ppi': [ppi],
                'is_ips': [is_ips],
                'is_touchscreen': [is_touchscreen],
                'Cpu_name': [cpu_name],
                'CPU_Speed_GHz': [cpu_speed],
                'SSD_GB': [ssd_gb],
                'HDD_GB': [hdd_gb],
                'gpu_brand': [gpu_brand]
            })
            
            print("User input data:")
            print(user_info_dataframe)
            
            # Load the model
            model = load_object(PRODUCTION_MODEL_PATH)
            
            # Make prediction
            laptop_price = model.predict_user_info(user_info_dataframe)[0]
            
            print(f"Predicted price: ${laptop_price:.2f}")
            
            # Return the prediction as JSON
            return jsonify({
                'laptop_price': round(laptop_price, 2),
                'price': round(laptop_price, 2),
                'prediction': round(laptop_price, 2)
            })
            
        except Exception as e:
            print("Error during prediction:", str(e))
            return jsonify({"error": str(e)}), 500

     
if __name__ == '__main__':
    app.run(debug = True)