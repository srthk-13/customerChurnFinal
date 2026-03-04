# 📉 Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to churn, built with a trained classification model and deployed via an interactive web application.

---

## 🚀 Demo

Link- https://customerchurnfinal-nbvfr3qlitzurp8zf32zjw.streamlit.app/

---

## 📁 Project Structure

```
customerChurnFinal/
├── app.py                  # Web application (Flask/Streamlit)
├── model.ipynb             # Jupyter notebook for EDA, training & evaluation
├── churn_model.pkl         # Serialized trained ML model
├── model_columns.pkl       # Saved feature columns for inference
├── customerChurn.csv       # Dataset used for training
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🧠 How It Works

1. **Data** — `customerChurn.csv` contains customer attributes (demographics, account info, usage patterns, etc.)
2. **Model Training** — `model.ipynb` covers data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation
3. **Inference** — The trained model (`churn_model.pkl`) and feature schema (`model_columns.pkl`) are loaded by `app.py` to serve predictions
4. **Web App** — Users input customer details via the UI and receive a churn prediction in real time

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/srthk-13/customerChurnFinal.git
cd customerChurnFinal
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Then open your browser and go to `http://localhost:5000` (or the port shown in the terminal).

---

## 📊 Model

- **Algorithm:** Classification model (e.g., Logistic Regression / Random Forest / XGBoost)
- **Input Features:** Customer demographics, contract type, tenure, charges, services subscribed, etc.
- **Output:** Binary prediction — `Churn` or `No Churn`
- **Serialization:** Model saved using `pickle` for fast loading during inference

---

## 📦 Dependencies

Key libraries used (see `requirements.txt` for full list):

- `pandas` — Data manipulation
- `scikit-learn` — Model training & preprocessing
- `flask` / `streamlit` — Web application framework
- `numpy` — Numerical operations
- `pickle` — Model serialization

---

## 📌 Usage

1. Launch the app using `python app.py`
2. Fill in the customer information fields in the UI
3. Click **Predict** to see whether the customer is likely to churn
4. Use the result to drive retention strategies

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
