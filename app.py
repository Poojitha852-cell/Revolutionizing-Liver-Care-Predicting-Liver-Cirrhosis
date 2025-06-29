from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('flask/rf_acc_68.pkl', 'rb'))
scaler = pickle.load(open('flask/normalizer.pkl', 'rb'))

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

@app.route('/')
def home():
    return render_template('assets/forms/index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()

        age = float(form_data['feature1'])
        gender = form_data['feature2']
        duration = float(form_data['feature3'])
        quantity = float(form_data['feature4'])
        hep_b = form_data['feature5']
        diabetes = form_data['feature6']
        obesity = form_data['feature7']
        hemoglobin = float(form_data['feature8'])
        platelet = float(form_data['feature9'])
        usg = form_data['feature10']

        # Encode
        enc_vals = encode_input(gender, hep_b, diabetes, obesity, usg)
        input_features = [age, enc_vals[0], duration, quantity,
                          enc_vals[1], enc_vals[2], enc_vals[3],
                          hemoglobin, platelet, enc_vals[4]]
        input_scaled = scaler.transform([input_features])
        prediction = model.predict(input_scaled)[0]

        result = (
            "🟢 Patient likely does NOT have liver cirrhosis."
            if prediction == 0
            else "🔴 Patient is likely suffering from liver cirrhosis."
        )

        return render_template('assets/forms/index.html', prediction_text=result, previous_data=form_data)

    except Exception as e:
        return render_template('assets/forms/index.html', prediction_text=f"❌ Error: {str(e)}", previous_data=request.form)

if __name__ == '__main__':
    app.run(debug=True)
