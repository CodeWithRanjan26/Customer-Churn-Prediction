# 📊 Customer Churn Prediction using Machine Learning & MySQL

Customer churn is one of the most critical challenges faced by subscription-based and service-oriented businesses. This project focuses on predicting customer churn using machine learning techniques, enabling businesses to identify high-risk customers early and take proactive retention actions.

The project demonstrates an end-to-end data science workflow, starting from database design and data extraction using MySQL, to data preprocessing, feature engineering, model training, evaluation, and business insight generation using Python.

---

## 🚀 Project Highlights

- End-to-end **Customer Churn Prediction System**
- Real-world **MySQL database integration**
- Feature engineering for meaningful insights
- Multiple ML models for comparison
- Business-focused interpretation of results
- Clean, modular, and production-ready code

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Database:** MySQL  
- **Libraries & Tools:**  
  - Pandas, NumPy  
  - Scikit-learn  
  - Matplotlib, Seaborn  
  - MySQL Connector  
- **IDE:** PyCharm  
- **Version Control:** Git & GitHub  

---

## 🗄️ Database Design

The project uses a relational MySQL database with the following tables:

- **customers** – demographic details and churn label  
- **services** – tenure, internet service, contract type  
- **billing** – charges and payment method  

Data is generated and stored in MySQL, then extracted using SQL joins for analysis and modeling.

---

## ⚙️ Data Processing & Feature Engineering

Key preprocessing steps include:
- Handling missing and inconsistent values
- Encoding categorical variables
- Scaling numerical features
- Creating engineered features such as:
  - Charges per month  
  - Tenure groups  
  - Monthly contract indicator  

These steps significantly improve model performance and interpretability.

---

## 🤖 Machine Learning Models

The following models are trained and evaluated:

- **Random Forest Classifier**
- **Logistic Regression**

Evaluation metrics:
- Accuracy
- ROC-AUC Score
- Confusion Matrix
- Classification Report

Random Forest performs best in identifying churn-prone customers.

---

## 📈 Visualizations

The project includes insightful visualizations such as:
- Churn distribution
- Churn rate by contract type
- Tenure distribution
- ROC curve
- Feature importance ranking

All visual outputs are saved as images for easy analysis and reporting.

---

## 🚨 High-Risk Customer Identification

Based on predicted churn probabilities, customers are classified into:
- Low Risk
- Medium Risk
- High Risk

This helps businesses focus retention strategies on the most vulnerable customers.

---

## Dashboard ScreenShort
<img width="1112" height="496" alt="image" src="https://github.com/user-attachments/assets/ce0f8f4f-aa77-4a84-a417-e5b462246461" />


## 📂 Project Structure


Customer-Churn-Prediction/
│
├── Prediction.py
├── main.py
├── churn_analysis.png
├── churn_analysis_mysql.png
├── .gitignore
├── LICENSE
└── README.md




---

## 📌 Business Value

- Helps reduce customer attrition
- Enables data-driven retention strategies
- Improves customer lifetime value
- Provides actionable insights for decision-makers

---

## 🙌 Conclusion

This project showcases practical implementation of machine learning for business problems, combining database management, data analytics, and predictive modeling. It reflects real-world data science workflows and demonstrates strong analytical and engineering skills.

---

## 👤 Author

**Ranjan Yadav**  
Aspiring Data Analyst | Machine Learning Enthusiast  

