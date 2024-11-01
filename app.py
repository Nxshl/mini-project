# app.py

from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the form
    input_data = [
        float(request.form['Safety_Score']),
        float(request.form['Days_Since_Inspection']),
        float(request.form['Total_Safety_Complaints']),
        float(request.form['Control_Metric']),
        float(request.form['Turbulence_In_gforces']),
        float(request.form['Cabin_Temperature']),
        int(request.form['Accident_Type_Code']),
        float(request.form['Max_Elevation']),
        float(request.form['Violations']),
        float(request.form['Adverse_Weather_Metric'])
    ]
    
    # Convert input data to numpy array and reshape for prediction
    input_data = np.array(input_data).reshape(1, -1)
    
    # Predict severity using the pre-trained model
    prediction = model.predict(input_data)
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
