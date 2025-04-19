from flask import request, render_template, Flask
import numpy as np
import pandas as pd
import pickle


#loading model
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        Area = request.form['area']
        Item = request.form['item']
        Year = request.form['year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']

        features = np.array([[Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]])
        transformed = preprocessor.transform(features)
        predicted_value = dtr.predict(transformed).reshape(1, -1)

        return render_template('index.html',predicted_value=predicted_value)



if __name__ == '__main__':
    app.run(debug=True)