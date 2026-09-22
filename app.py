# Importing core libraries for data handling and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Importing machine learning modules from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuring the main Streamlit application layout
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# Applying custom CSS for clean metric cards and headers
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

# Displaying main dashboard title and project author
st.markdown('<div class="main-header">📊 Telco Customer Churn & Retention Dashboard</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">Author: Jyoti Thakur | Predictive Analytics & Customer Retention System</div>', unsafe_allow_html=True)

# Defining cached data loading function to optimize performance
@st.cache_data
def load_data():

    # Loading the dataset directly from remote source
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    # Cleaning TotalCharges column by converting text values to numbers
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Imputing missing values in TotalCharges using direct assignment
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    return df

# Executing data load function
df = load_data()

# Creating navigation tabs for structured presentation
tab1, tab2, tab3 = st.tabs(["📌 Overview & Data", "📈 EDA Insights", "🔮 Live Churn Predictor"])

# Section for Tab 1: Dataset Overview and Key Metrics
with tab1:

    st.subheader("Key Business Metrics")

    # Calculating high level KPI numbers
    total_customers = len(df)
    churn_count = (df['Churn'] == 'Yes').sum()
    churn_rate = (churn_count / total_customers) * 100
    avg_monthly = df['MonthlyCharges'].mean()

    # Rendering four metric cards across columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Churned", f"{churn_count:,}")
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("Avg Monthly Bill", f"${avg_monthly:.2f}")

    st.markdown("---")

    st.subheader("Dataset Preview")

    # Showing raw dataset sample table
    st.dataframe(df.head(10), use_container_width=True)

# Section for Tab 2: Exploratory Data Analysis
with tab2:

    st.subheader("Exploratory Data Analysis")

    sns.set_theme(style="whitegrid")

    col1, col2 = st.columns(2)

    with col1:

        # Chart 1: Customer Churn Ratio
        st.write("##### 1. Overall Churn Ratio")
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        sns.countplot(data=df, x='Churn', hue='Churn', palette=['#10B981', '#EF4444'], legend=False, ax=ax1)
        ax1.set_title("Customer Churn Distribution")
        st.pyplot(fig1, use_container_width=True)
        st.info("💡 **Observation:** Approximately 26.5% of total customers have churned.")

    with col2:

        # Chart 2: Contract Type Impact
        st.write("##### 2. Churn Rate by Contract Type")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        sns.countplot(data=df, x='Contract', hue='Churn', palette=['#10B981', '#EF4444'], ax=ax2)
        ax2.set_title("Contract Type vs Churn")
        st.pyplot(fig2, use_container_width=True)
        st.info("💡 **Observation:** Month-to-month contract holders show the highest churn tendency.")

    col3, col4 = st.columns(2)

    with col3:

        # Chart 3: Tenure Distribution
        st.write("##### 3. Customer Tenure Distribution")
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, palette=['#10B981', '#EF4444'], ax=ax3)
        ax3.set_title("Tenure (Months) Density by Churn")
        st.pyplot(fig3, use_container_width=True)
        st.info("💡 **Observation:** New customers with 0 to 12 months tenure exhibit the highest churn risk.")

    with col4:

        # Chart 4: Internet Service Type Impact
        st.write("##### 4. Internet Service Type vs Churn")
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        sns.countplot(data=df, x='InternetService', hue='Churn', palette=['#10B981', '#EF4444'], ax=ax4)
        ax4.set_title("Internet Service Type vs Churn")
        st.pyplot(fig4, use_container_width=True)
        st.info("💡 **Observation:** Fiber optic users display higher churn compared to DSL users.")

    # Chart 5: Monthly Charges BoxPlot
    st.write("##### 5. Monthly Charges vs Churn Status")
    fig5, ax5 = plt.subplots(figsize=(8, 3))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', hue='Churn', palette=['#10B981', '#EF4444'], legend=False, ax=ax5)
    ax5.set_title("Monthly Charges ($) vs Churn")
    st.pyplot(fig5, use_container_width=True)
    st.info("💡 **Observation:** Higher monthly billing values correlate with increased customer churn likelihood.")

# Section for Tab 3: Machine Learning Model and Live Risk Evaluation
df_ml = df.drop(columns=['customerID']).copy()

# Converting string categories into numbers using Label Encoder
encoders = {}
object_cols = df_ml.select_dtypes(include=['object', 'string']).columns

for col in object_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    encoders[col] = le

# Splitting feature columns and target variable
X = df_ml.drop(columns=['Churn'])
y = df_ml['Churn']

# Performing train test split with an 80 to 20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculating overall model testing accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

with tab3:

    st.subheader("Machine Learning Performance")
    st.success(f"🎯 **Model Testing Accuracy:** {acc * 100:.2f}%")

    st.markdown("---")

    st.subheader("🔮 Live Churn Risk Calculator")

    # Collecting customer input values for prediction
    c1, c2, c3 = st.columns(3)

    with c1:
        in_tenure = st.slider("Tenure (Months)", 1, 72, 12)
        in_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        in_internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with c2:
        in_monthly = st.number_input("Monthly Charges ($)", value=70.0)
        in_total = st.number_input("Total Charges ($)", value=in_tenure * in_monthly)
        in_payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with c3:
        in_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        in_tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        in_paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    # Evaluating prediction upon button click
    if st.button("🚀 Calculate Risk", type="primary"):

        # Structuring input payload matching training format
        sample_input = {
            'gender': 0, 'SeniorCitizen': 0, 'Partner': 0, 'Dependents': 0,
            'tenure': in_tenure,
            'PhoneService': 1, 'MultipleLines': 0,
            'InternetService': encoders['InternetService'].transform([in_internet])[0],
            'OnlineSecurity': encoders['OnlineSecurity'].transform([in_security])[0],
            'OnlineBackup': 0, 'DeviceProtection': 0,
            'TechSupport': encoders['TechSupport'].transform([in_tech])[0],
            'StreamingTV': 0, 'StreamingMovies': 0,
            'Contract': encoders['Contract'].transform([in_contract])[0],
            'PaperlessBilling': encoders['PaperlessBilling'].transform([in_paperless])[0],
            'PaymentMethod': encoders['PaymentMethod'].transform([in_payment])[0],
            'MonthlyCharges': in_monthly,
            'TotalCharges': in_total
        }

        input_df = pd.DataFrame([sample_input])

        # Estimating probability of churn
        churn_prob = model.predict_proba(input_df)[0][1] * 100

        st.markdown("### **Prediction Result:**")

        if churn_prob > 50:
            st.error(f"⚠️ **High Churn Risk Detected!** Probability: **{churn_prob:.1f}%**")
            st.warning("📌 **Recommended Action:** Offer long-term contract discounts and technical support add-ons.")
        else:
            st.success(f"✅ **Low Churn Risk!** Probability: **{churn_prob:.1f}%**")
            st.info("📌 **Recommended Action:** Customer profile is stable. Target with premium service upgrades.")