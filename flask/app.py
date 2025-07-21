from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load models using paths relative to current file
model_path = os.path.join(os.path.dirname(__file__), 'rf_acc_68.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'normalizer.pkl')

model = joblib.load(model_path)
normalizer = joblib.load(scaler_path)

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

        # Convert categorical values to numerical
        gender = 1 if data[1].lower() == 'male' else 0
        hepatitis = 1 if data[4].lower() == 'yes' else 0
        diabetes = 1 if data[5].lower() == 'yes' else 0
        obesity = 1 if data[6].lower() == 'yes' else 0
        usg = 1 if data[9].lower() == 'diffuse' else 0

        features = np.array([
            float(data[0]),   # age
            gender,
            float(data[2]),   # duration
            float(data[3]),   # quantity
            hepatitis,
            diabetes,
            obesity,
            float(data[7]),   # hemoglobin
            float(data[8]),   # platelet
            usg
        ]).reshape(1, -1)

        # Normalize and predict
        normalized_input = normalizer.transform(features)
        prediction = model.predict(normalized_input)[0]

        result = (
            "🟢 You are NOT likely to have Liver Cirrhosis"
            if prediction == 0
            else "🔴 You are likely to have Liver Cirrhosis"
        )

        return render_template('index.html', prediction_text=result, previous_data=request.form)

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error during prediction: {e}")

if __name__ == '__main__':
    app.run(debug=True)


