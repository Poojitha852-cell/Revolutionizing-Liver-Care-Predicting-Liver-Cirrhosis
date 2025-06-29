# Revolutionizing Liver Care: Predicting Liver Cirrhosis Using Advanced Machine Learning Techniques

## 🔖 Overview

This project is a web-based liver cirrhosis prediction application powered by machine learning and a neon-themed UI using Flask. It takes clinical features as input and predicts whether a patient is likely suffering from liver cirrhosis.

---

## 📅 Project Structure

```
LiverCirrhosisPrediction/
├── app.py
├── models/
│   ├── rf_acc_68.pkl
│   └── normalizer.pkl
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

---

## 🚀 Features

* User-friendly web interface (neon style)
* Takes 10 clinical input features
* Displays prediction visually and keeps previous input data
* Powered by Random Forest classifier
* Fully responsive and interactive UI

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/LiverCirrhosisPrediction.git
cd LiverCirrhosisPrediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🔬 Dataset

* File: `dataset/HealthCareData.xlsx`
* Includes features such as Age, Gender, Alcohol Consumption, Hemoglobin, Platelet Count, Hepatitis B, etc.

---

## 🤖 Model Training

* Model: `RandomForestClassifier`
* Training script: `training/liver_model_train.py`
* Output:

  * `models/rf_acc_68.pkl` (trained model)
  * `models/normalizer.pkl` (StandardScaler)

---

## 🌐 Usage

### Run the Flask App:

```bash
python app.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📈 Input Features

* Age
* Gender
* Duration of Alcohol Consumption
* Quantity of Alcohol Consumption
* Hepatitis B Infection
* Diabetes
* Obesity
* Hemoglobin (g/dl)
* Platelet Count (lakhs/mm)
* USG Abdomen (Diffuse/Normal)

---

## 🔹 Output

* 🟢 **Patient likely does NOT have liver cirrhosis**
* 🔴 **Patient is likely suffering from liver cirrhosis**

---

## 🏓 Screenshots

> Add UI screenshots here in your GitHub repo.

---

## 📥 Deployment

* Local deployment with Flask
* Can be extended to deploy on platforms like Heroku, Render, or PythonAnywhere

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 📚 References

* scikit-learn documentation
* Flask official documentation
* Dataset (provide your source or citation)
* CSS neon UI inspiration from CodePen and CSS-Tricks
