# Boston-Production-
# 🏠 Boston House Price Prediction

An end-to-end machine learning project that predicts the median house value (MEDV) for towns in the Boston housing dataset, using structural and socio-economic features. The trained model is deployed as an interactive **Streamlit** web app.

---

## 🎯 Objective

Predict `MEDV` — the median value of owner-occupied homes (in $1000s) — using 11 input features describing housing and neighborhood characteristics.

---

## 📊 Dataset

The dataset contains **333 records** and **12 columns** (11 features + target), with no missing values and no duplicate rows.

| Column | Description |
|---|---|
| ID | Record identifier (used here as a numeric feature) |
| indus | Proportion of non-retail business acres per town |
| chas | Charles River dummy variable (1 if tract bounds river, else 0) |
| nox | Nitric oxide concentration (parts per 10 million) |
| rm | Average number of rooms per dwelling |
| age | Proportion of owner-occupied units built before 1940 |
| dis | Weighted distance to five Boston employment centres |
| rad | Index of accessibility to radial highways |
| tax | Full-value property tax rate per $10,000 |
| ptratio | Pupil-teacher ratio by town |
| lstat | % lower status of the population |
| **medv** (target) | Median value of owner-occupied homes, in $1000s |

---

## 🧹 Data Preparation & EDA

- Checked for missing values and duplicates — none found
- Correlation heatmap, histograms, scatter plots, and pairplots used to explore relationships
- `rm` shows a strong positive relationship with `medv`; `lstat` shows a strong negative relationship
- Outlier analysis performed on `rm`, `lstat`, `dis`, and `ptratio` using the IQR method

---

## 🤖 Models Trained & Compared

| Model | MAE | MSE | RMSE | R² Score |
|---|---|---|---|---|
| Decision Tree (default) | 4.73 | 47.10 | 6.86 | 0.620 |
| Decision Tree (tuned, GridSearchCV) | 4.16 | 48.87 | 6.99 | 0.606 |
| KNN (k=3, default) | 4.85 | 74.13 | 8.61 | 0.402 |
| KNN (tuned, GridSearchCV) | 5.10 | 82.34 | 9.07 | 0.336 |
| **Random Forest (default)** | **3.57** | **45.28** | **6.73** | **0.635** |
| Random Forest (tuned, GridSearchCV) | 3.53 | 46.26 | 6.80 | 0.627 |

Random Forest (default) gave the best test-set performance and was selected as the final model, saved as `Boston_Project_Model.pkl`.

**Most important features:** `rm` (avg. rooms) and `lstat` (% lower status population) — together accounting for ~86% of feature importance.

---

## 📁 Project Structure

```
├── Boston_clean_file_1_.ipynb    # Data cleaning & EDA
├── Boston_file_2_.ipynb          # Model training, tuning & comparison
├── Prediction.ipynb              # Sample prediction using the saved model
├── Boston_Project_Model.pkl      # Final trained model (Random Forest)
├── app.py                        # Streamlit web app
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Running the App Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`, where you can enter property details and get an instant MEDV prediction.

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **scikit-learn** — DecisionTreeRegressor, KNeighborsRegressor, RandomForestRegressor, GridSearchCV
- **matplotlib, seaborn** — visualization
- **joblib** — model persistence
- **Streamlit** — web app deployment

---

## 📌 Future Improvements

- Try gradient boosting models (XGBoost, LightGBM) for potentially better R²
- Add cross-validated feature selection to reduce noise from low-importance features
- Add confidence intervals around predictions instead of a single point estimate

---

## 📄 License

This project is open-source and available for learning/demo purposes.
