from laptopPrice.exception import LaptopException
from flask import Flask, render_template, request, jsonify
from laptopPrice.pipeline.prediction_pipeline import CustomData , PredictPipeline


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
            
            customData = CustomData(data_dict = data)             
            
            # Make prediction
            prediction_pipeline = PredictPipeline()
            predictions_dict = prediction_pipeline.predict(custom_data = customData)
            
            # Return the prediction as JSON
            return jsonify(predictions_dict)
            
        except Exception as e:
            print("Error during prediction:", str(e))
            return jsonify({"error": str(e)}), 500

     
if __name__ == '__main__':
    app.run(debug = True)