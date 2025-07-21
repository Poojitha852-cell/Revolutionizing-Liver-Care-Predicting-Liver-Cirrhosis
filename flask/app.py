from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'rf_acc_68.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'normalizer.pkl')

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Encoding helper
def encode_input(gender, hep_b, diabetes, obesity, usg):
    gender_map = {"male": 1, "female": 0}
    binary_map = {"yes": 1, "no": 0}
    usg_map = {"diffuse": 1, "normal": 0}

    return [
        gender_map.get(gender.lower(), 0),
        binary_map.get(hep_b.lower(), 0),
        binary_map.get(diabetes.lower(), 0),
        binary_map.get(obesity.lower(), 0),
        usg_map.get(usg.lower(), 0)
    ]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    if request.method == 'POST':
        try:
            # Get and parse form data
            age = float(request.form['feature1'])
            gender = request.form['feature2']
            duration = float(request.form['feature3'])
            quantity = float(request.form['feature4'])
            hep_b = request.form['feature5']
            diabetes = request.form['feature6']
            obesity = request.form['feature7']
            hemoglobin = float(request.form['feature8'])
            platelet = float(request.form['feature9'])
            usg = request.form['feature10']

            encoded = encode_input(gender, hep_b, diabetes, obesity, usg)

            features = [
                age, encoded[0], duration, quantity, encoded[1],
                encoded[2], encoded[3], hemoglobin, platelet, encoded[4]
            ]

            scaled_input = scaler.transform([features])
            prediction = model.predict(scaled_input)[0]

            prediction_text = (
                "🟢 Patient likely does NOT have liver cirrhosis."
                if prediction == 0 else
                "🔴 Patient is likely suffering from liver cirrhosis."
            )

        except Exception as e:
            prediction_text = f"❌ Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
