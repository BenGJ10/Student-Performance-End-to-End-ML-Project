from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = request.form.get('reading_score'),
            writing_score = request.form.get('writing_score')
        )
        prediction_data = data.get_data_as_dataframe()
        print(prediction_data)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(prediction_data)
        return render_template('home.html', results = results[0])

if __name__=="__main__":      
    app.run(host = "0.0.0.0", port=80)    