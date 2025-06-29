# 🧬 Liver Cirrhosis Prediction using Machine Learning

This project predicts the likelihood of **liver cirrhosis** in a patient based on clinical and diagnostic inputs using a trained machine learning model. The application is built using **Flask** with a **neon-themed user interface**.

---

## 📁 Project Structure

```
Revolutionizing-Liver-Care-Predicting-Liver-Cirrhosis/
├── data/
│   └── HealthCareData.xlsx
├── documentation/
│   └── README.md
├── flask/
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   ├── inner-page.html
│   │   └── portfolio-details.html
│   ├── app.py
│   ├── rf_acc_68.pkl
│   └── normalizer.pkl
├── training/
│   └── liver_model_train.py
├── requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Revolutionizing-Liver-Care-Predicting-Liver-Cirrhosis.git
cd Revolutionizing-Liver-Care-Predicting-Liver-Cirrhosis
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Linux/macOS
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Run the Flask Application

```bash
cd flask
python app.py
```

Visit **`http://127.0.0.1:5000/`** in your browser to access the web app.

---

## 🧠 Model Details

* Model: **Random Forest Classifier**
* Accuracy: \~68%
* Preprocessing: Features normalized using a saved `normalizer.pkl` object
* Encodes categorical values like Gender, Diabetes, Hepatitis B, etc.

---

## 🎨 User Interface (Neon Theme)

* Full-page neon UI
* Styled using `static/style.css`
* Inputs retained after prediction
* Prediction result highlighted with **color-coded alert**

---

## 📦 Requirements

Dependencies are listed in `requirements.txt`. Major packages include:

* `Flask`
* `numpy`
* `scikit-learn`
* `pandas`

Install using:

```bash
pip install -r requirements.txt
```

---

## 📸 Screenshots

![Uploading line_follower_robot.png…]()


---

## 📑 Notes

* The `rf_acc_68.pkl` file contains the trained Random Forest model.
* `normalizer.pkl` contains the fitted scaler used during model training.
* `liver_model_train.py` (in training folder) contains the training pipeline.
* Inputs are not reset after submitting to ensure user convenience.
* The HTML and CSS are placed in Flask’s conventional `templates` and `static` folders.

---

## 🙋‍♀️ Author

* Developed as part of an AI internship project.
* Feel free to fork or suggest improvements!

