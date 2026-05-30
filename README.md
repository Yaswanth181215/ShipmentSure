# 🚚 ShipmentSure: Predicting On-Time Delivery Using Supplier Data

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![ML](https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost-orange)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

## 📌 Overview
ShipmentSure is an end-to-end machine learning project that predicts whether a shipment will arrive **on time or be delayed**, based on supplier and order-level features. Built for logistics and manufacturing firms to improve procurement planning and delivery reliability.

---

## 🎯 Problem Statement
Late deliveries cost businesses millions every year. By leveraging historical order data, ShipmentSure helps operations teams **proactively identify at-risk shipments** before they happen.

---

## 🗂️ Project Structure
```
ShipmentSure/
│
├── data/                   # Raw and processed datasets
├── models/                 # Saved trained models (.pkl)
├── notebooks/              # Jupyter notebooks for EDA & modeling
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
├── src/                    # Source code modules
│   ├── preprocess.py       # Data cleaning & feature engineering
│   ├── train.py            # Model training & evaluation
│   └── predict.py          # Prediction pipeline
├── app.py                  # Streamlit web application
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🧠 ML Pipeline
1. **EDA** — Univariate/bivariate analysis, class imbalance check
2. **Preprocessing** — Encoding, normalization, feature engineering
3. **Modeling** — Logistic Regression, Random Forest, XGBoost
4. **Evaluation** — Accuracy, F1, ROC-AUC, Confusion Matrix
5. **Deployment** — Streamlit web app with real-time predictions

---

## 📊 Dataset
- **Source:** Kaggle – Supply Chain Logistics Dataset
- **Target:** `Reached.on.Time_Y.N` (1 = On Time, 0 = Delayed)
- **Features:** 10 features including shipment mode, product importance, discount, weight, calls, rating, etc.

---

## 🏆 Model Performance (on test set)

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~67% | ~0.65 | ~0.70 |
| Random Forest | ~69% | ~0.68 | ~0.74 |
| **XGBoost** | **~71%** | **~0.70** | **~0.76** |

---

## ⚙️ Tech Stack
| Component | Tools |
|---|---|
| Language | Python 3.9+ |
| Data Handling | Pandas, NumPy |
| Visualization | Seaborn, Matplotlib |
| Modeling | scikit-learn, XGBoost |
| Deployment | Streamlit |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ShipmentSure.git
cd ShipmentSure

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place in data/ folder

# 4. Train the model
python src/train.py

# 5. Launch the web app
streamlit run app.py
```

---

## 👨‍💻 Author
Built as part of Infosys HackWithInfy 2026 preparation.
