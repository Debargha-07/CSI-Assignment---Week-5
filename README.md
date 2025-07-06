# CSI-Assignment---Week-5

# 🏠 House Price Prediction — Preprocessing & Feature Engineering
## Author : Debargha Karmakar
Welcome to my Celebal Summer Internship Assignment (Week 5)!  
This project focuses on **data preprocessing**, **EDA**, and **feature engineering** for the House Price Prediction problem using the Kaggle dataset.

---

## 🚀 Project Objective

To clean, transform, and prepare housing data for machine learning models through:

- 🔍 Exploratory Data Analysis (EDA)
- 🧼 Data Cleaning & Missing Value Handling
- 🧠 Feature Engineering
- 🔢 Skew Correction
- 🎭 One-Hot Encoding

---

## 📊 Dataset

- **Source:** [Kaggle - House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- `train.csv`: 1460 samples with target variable `SalePrice`
- `test.csv`: 1459 samples without the target
- **Target:** `SalePrice` (continuous variable)

---


## 📊 Exploratory Data Analysis

Visualizations saved in `.png` format:

- 📈 Sale Price Distribution  
- 🔥 Top Correlated Features  
- 🧼 Missing Value Chart  

All visualizations generated during EDA (like missing value charts, correlation heatmaps, and distribution plots) are included in a separate PDF report.

📥 Download All Plots Report: EDA_Plots.pdf


---

## 🛠️ Feature Engineering Highlights

| Feature        | Description                          |
|----------------|--------------------------------------|
| `TotalSF`      | Total finished square footage        |
| `TotalBath`    | Total weighted bathrooms             |
| `HouseAge`     | Age of house at sale                 |
| `RemodelAge`   | Years since last remodel             |
| `GarageAge`    | Age of garage at sale                |

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `X_train_preprocessed.csv` | Cleaned, encoded training data |
| `X_test_preprocessed.csv`  | Processed test data |
| `y_train.csv`              | Target variable |
| `.png` files               | Saved plots from EDA |

---

## 📦 Technologies Used

- Python 3.11  
- Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn  
- VS Code

---

## 📎 How to Run

1. Clone this repo  
2. Place `train.csv` and `test.csv` in the root folder  
3. Run the script:

```bash
python preprocessing.py

