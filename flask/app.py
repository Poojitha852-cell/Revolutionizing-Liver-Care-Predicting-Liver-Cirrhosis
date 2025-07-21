from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load models
model = joblib.load('models/rf_acc_68.pkl')
normalizer = joblib.load('models/normalizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            request.form['feature1'],
            request.form['feature2'],
            request.form['feature3'],
            request.form['feature4'],
            request.form['feature5'],
            request.form['feature6'],
            request.form['feature7'],
            request.form['feature8'],
            request.form['feature9'],
            request.form['feature10'],
        ]

        # Convert gender and categorical to numbers
        gender = 1 if data[1].lower() == 'male' else 0
        hepatitis = 1 if data[4].lower() == 'yes' else 0
        diabetes = 1 if data[5].lower() == 'yes' else 0
        obesity = 1 if data[6].lower() == 'yes' else 0
        usg = 1 if data[9].lower() == 'diffuse' else 0

        features = np.array([
            float(data[0]),
            gender,
            float(data[2]),
            float(data[3]),
            hepatitis,
            diabetes,
            obesity,
            float(data[7]),
            float(data[8]),
            usg
        ]).reshape(1, -1)

        normalized_input = normalizer.transform(features)
        prediction = model.predict(normalized_input)[0]

        result = "You are NOT likely to have Liver Cirrhosis" if prediction == 0 else "You are likely to have Liver Cirrhosis"
        return render_template('index.html', prediction_text=result, previous_data=request.form)
    except Exception as e:
        return f"Error during prediction: {e}"

