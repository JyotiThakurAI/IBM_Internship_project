# Telco Customer Churn & Retention Analytics

**Author:** Jyoti Thakur

**Technology Stack:** Python, Pandas, NumPy, Scikit-learn, Streamlit, Seaborn, Matplotlib

---

## Project Overview

This project provides an end-to-end Data Analytics and Machine Learning pipeline designed to analyze and predict customer churn in the telecommunications sector.

The project analyzes customer demographics, account information, service usage, and other customer-related attributes to identify patterns associated with customer churn. A Machine Learning model is then used to predict whether a customer is likely to churn.

An interactive Streamlit dashboard is also developed to allow users to explore the data and generate churn predictions.

---

## Dataset

The project uses the **Telco Customer Churn dataset**.

**Dataset Source:** Kaggle – Telco Customer Churn
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset file used in this project is included in the project folder:

`Telco-Customer-Churn.csv`

---

## Key Deliverables

* **Data Cleaning & Engineering:** Handled missing values, converted data types, and prepared categorical features for Machine Learning.
* **Exploratory Data Analysis:** Analyzed customer characteristics and factors related to customer churn using visualizations.
* **Predictive Machine Learning:** Trained a Random Forest Classifier for customer churn prediction.
* **Interactive Dashboard:** Built a Streamlit web application (`app.py`) for customer information input and churn prediction.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Jupyter Notebook

---

## Project Structure

```text
IBM_Internship/
│
├── JyotiThakur_TelcoChurn.ipynb
├── app.py
├── requirements.txt
├── README.md
├── JyotiThakur_ProjectReport.docx
└── Telco-Customer-Churn.csv
```

---

## Setup & Running Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/JyotiThakurAI/IBM_Internship_project.git
```

### 2. Navigate to the Project Folder

```bash
cd IBM_Internship_project
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser. If it does not open automatically, use the local URL displayed in the terminal.

---

## Machine Learning Model

The project uses a **Random Forest Classifier** to predict customer churn based on customer and service-related features.

The model achieved approximately **80% accuracy** on the evaluated dataset.

---

## Dashboard Features

The Streamlit application provides:

* Customer information input
* Customer churn prediction
* Data exploration
* Prediction results
* Interactive user interface

---

## Project Report

Detailed project methodology, analysis, Machine Learning implementation, results, and conclusions are documented in:

`JyotiThakur_ProjectReport.docx`

---

## Author

**Jyoti Thakur**

BCA Student | Aspiring Data Analyst
