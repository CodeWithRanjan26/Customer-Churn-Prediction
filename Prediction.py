
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
import mysql.connector
from mysql.connector import Error
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("CUSTOMER CHURN PREDICTION - MYSQL DATABASE")
print("=" * 70)

# ===========================================================================
# STEP 1: MYSQL DATABASE CONNECTION
# ===========================================================================
print("\n[STEP 1] Connecting to MySQL Database...")

# Enter your MySQL settings here
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'customer_churn',
    'auth_plugin': 'mysql_native_password'
}


try:
    # Create MySQL connection
    connection = mysql.connector.connect(**db_config)

    if connection.is_connected():
        print(f"✓ Successfully connected to MySQL database: {db_config['database']}")
        cursor = connection.cursor()

        # Check database version
        cursor.execute("SELECT VERSION()")
        db_version = cursor.fetchone()
        print(f"  MySQL Version: {db_version[0]}")

except Error as e:
    print(f"❌ Error connecting to MySQL: {e}")
    print("\nPlease ensure that:")
    print("1. MySQL server is running")
    print("2. Database 'customer_churn' exists")
    print("3. Username and Password are correct")
    exit()

# ===========================================================================
# STEP 2: CREATE DATABASE TABLES (if they don't exist)
# ===========================================================================
print("\n[STEP 2] Creating database tables...")

create_tables_sql = """
-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id INT PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen INT,
    partner VARCHAR(3),
    dependents VARCHAR(3),
    churn INT
);

-- Services table
CREATE TABLE IF NOT EXISTS services (
    customer_id INT,
    tenure INT,
    phone_service VARCHAR(3),
    internet_service VARCHAR(20),
    contract_type VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Billing table
CREATE TABLE IF NOT EXISTS billing (
    customer_id INT,
    monthly_charges DECIMAL(10,2),
    total_charges DECIMAL(10,2),
    payment_method VARCHAR(30),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

# Create tables
for statement in create_tables_sql.split(';'):
    if statement.strip():
        try:
            cursor.execute(statement)
        except Error as e:
            print(f"Note: {e}")

connection.commit()
print("✓ Tables are ready")

# ===========================================================================
# STEP 3: INSERT SAMPLE DATA (if tables are empty)
# ===========================================================================
print("\n[STEP 3] Checking sample data...")

cursor.execute("SELECT COUNT(*) FROM customers")
customer_count = cursor.fetchone()[0]

if customer_count == 0:
    print("  Tables are empty, inserting sample data...")

    # Generate sample data
    np.random.seed(42)
    n_customers = 1000

    customers_data = []
    services_data = []
    billing_data = []

    for i in range(1, n_customers + 1):
        # Customer data
        customers_data.append((
            int(i),
            str(np.random.choice(['Male', 'Female'])),
            int(np.random.choice([0, 1], p=[0.85, 0.15])),
            str(np.random.choice(['Yes', 'No'])),
            str(np.random.choice(['Yes', 'No'], p=[0.7, 0.3])),
            int(np.random.choice([0, 1], p=[0.73, 0.27]))
        ))

        # Services data
        services_data.append((
            int(i),
            int(np.random.randint(1, 73)),
            str(np.random.choice(['Yes', 'No'], p=[0.9, 0.1])),
            str(np.random.choice(['DSL', 'Fiber optic', 'No'])),
            str(np.random.choice(
                ['Month-to-month', 'One year', 'Two year'],
                p=[0.55, 0.25, 0.20]
            ))
        ))

        # Billing data
        billing_data.append((
            int(i),
            float(round(np.random.uniform(18.0, 118.0), 2)),
            float(round(np.random.uniform(18.0, 8500.0), 2)),
            str(np.random.choice([
                'Electronic check',
                'Mailed check',
                'Bank transfer',
                'Credit card'
            ]))
        ))

    # Insert data
    cursor.executemany("""
        INSERT INTO customers (customer_id, gender, senior_citizen, partner, dependents, churn)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, customers_data)

    cursor.executemany("""
        INSERT INTO services (customer_id, tenure, phone_service, internet_service, contract_type)
        VALUES (%s, %s, %s, %s, %s)
    """, services_data)

    cursor.executemany("""
        INSERT INTO billing (customer_id, monthly_charges, total_charges, payment_method)
        VALUES (%s, %s, %s, %s)
    """, billing_data)

    connection.commit()
    print(f"✓ Inserted data for {n_customers} customers")
else:
    print(f"✓ Database already contains {customer_count} customers")

# ===========================================================================
# STEP 4: EXTRACT DATA USING SQL QUERY
# ===========================================================================
print("\n[STEP 4] Extracting data using SQL query...")

sql_query = """
SELECT 
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    s.tenure,
    s.phone_service,
    s.internet_service,
    s.contract_type,
    b.monthly_charges,
    b.total_charges,
    b.payment_method,
    c.churn
FROM customers c
JOIN services s ON c.customer_id = s.customer_id
JOIN billing b ON c.customer_id = b.customer_id
"""

# Load data with Pandas
df = pd.read_sql(sql_query, connection)

# Close connection
cursor.close()
connection.close()

print(f"✓ Data successfully extracted")
print(f"  - Shape: {df.shape}")
print(f"  - Columns: {list(df.columns)}")

# ===========================================================================
# STEP 5: DATA PREPROCESSING
# ===========================================================================
print("\n[STEP 5] Data preprocessing...")

df_processed = df.copy()
df_processed['total_charges'] = pd.to_numeric(df_processed['total_charges'], errors='coerce')
df_processed['total_charges'].fillna(df_processed['total_charges'].median(), inplace=True)

# Encode binary columns
binary_cols = ['gender', 'partner', 'dependents', 'phone_service']
for col in binary_cols:
    df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

# Encode categorical columns
label_encoders = {}
categorical_cols = ['internet_service', 'contract_type', 'payment_method']
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

print(f"✓ Encoded {len(binary_cols) + len(categorical_cols)} features")

# ===========================================================================
# STEP 6: FEATURE ENGINEERING
# ===========================================================================
print("\n[STEP 6] Feature engineering...")

df_processed['tenure_group'] = pd.cut(df_processed['tenure'],
                                      bins=[0, 12, 24, 48, 72],
                                      labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])
df_processed['charges_per_month'] = df_processed['total_charges'] / (df_processed['tenure'] + 1)
df_processed['is_monthly_contract'] = (df_processed['contract_type'] == 'Month-to-month').astype(int)
df_processed['tenure_group_encoded'] = LabelEncoder().fit_transform(df_processed['tenure_group'])

print("✓ Created 3 new features")

# ===========================================================================
# STEP 7: MODEL TRAINING
# ===========================================================================
print("\n[STEP 7] Model training...")

feature_columns = [
    'senior_citizen', 'partner', 'dependents', 'tenure', 'phone_service',
    'monthly_charges', 'total_charges', 'internet_service_encoded',
    'contract_type_encoded', 'payment_method_encoded', 'charges_per_month',
    'is_monthly_contract', 'tenure_group_encoded'
]

X = df_processed[feature_columns]
y = df_processed['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
print("✓ Random Forest trained")

lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
print("✓ Logistic Regression trained")

# ===========================================================================
# STEP 8: MODEL EVALUATION
# ===========================================================================
print("\n[STEP 8] Model evaluation...")

y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\n" + "=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
print("\n" + classification_report(y_test, y_pred_rf, target_names=['Not Churned', 'Churned']))

# ===========================================================================
# STEP 9: FEATURE IMPORTANCE
# ===========================================================================
print("\n[STEP 9] Feature importance analysis...")

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10).to_string(index=False))

# ===========================================================================
# STEP 10: VISUALIZATIONS
# ===========================================================================
print("\n[STEP 10] Creating visualizations...")

fig = plt.figure(figsize=(18, 12))

# Churn Distribution
ax1 = plt.subplot(3, 3, 1)
churn_counts = df['churn'].value_counts()
ax1.pie(churn_counts, labels=['Not Churned', 'Churned'], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
ax1.set_title('Churn Distribution', fontweight='bold')

# Churn by Contract
ax2 = plt.subplot(3, 3, 2)
df.groupby('contract_type')['churn'].mean().plot(kind='bar', ax=ax2, color='#3498db')
ax2.set_title('Churn Rate by Contract', fontweight='bold')
ax2.set_ylabel('Churn Rate')
ax2.tick_params(axis='x', rotation=45)

# Tenure Distribution
ax3 = plt.subplot(3, 3, 3)
df[df['churn'] == 0]['tenure'].hist(bins=30, alpha=0.5, label='Not Churned', ax=ax3, color='#2ecc71')
df[df['churn'] == 1]['tenure'].hist(bins=30, alpha=0.5, label='Churned', ax=ax3, color='#e74c3c')
ax3.set_title('Tenure Distribution', fontweight='bold')
ax3.legend()

# Confusion Matrix
ax4 = plt.subplot(3, 3, 4)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_title('Confusion Matrix', fontweight='bold')
ax4.set_ylabel('Actual')
ax4.set_xlabel('Predicted')

# ROC Curve
ax5 = plt.subplot(3, 3, 5)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_rf)
ax5.plot(fpr, tpr, label=f'ROC (AUC={auc(fpr, tpr):.2f})', color='#e74c3c', linewidth=2)
ax5.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax5.set_title('ROC Curve', fontweight='bold')
ax5.set_xlabel('False Positive Rate')
ax5.set_ylabel('True Positive Rate')
ax5.legend()

# Feature Importance
ax6 = plt.subplot(3, 3, 6)
top_10 = feature_importance.head(10)
ax6.barh(top_10['feature'], top_10['importance'], color='#9b59b6')
ax6.set_title('Feature Importance', fontweight='bold')
ax6.invert_yaxis()

plt.tight_layout()
plt.savefig('churn_analysis_mysql.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: churn_analysis_mysql.png")
plt.show()

# ===========================================================================
# STEP 11: IDENTIFY HIGH-RISK CUSTOMERS
# ===========================================================================
print("\n[STEP 11] Identifying high-risk customers...")

df_results = df_processed.copy()
df_results['churn_probability'] = rf_model.predict_proba(scaler.transform(df_processed[feature_columns]))[:, 1]
df_results['risk_level'] = pd.cut(df_results['churn_probability'],
                                  bins=[0, 0.3, 0.7, 1.0],
                                  labels=['Low', 'Medium', 'High'])

high_risk = df_results[df_results['risk_level'] == 'High']
print(f"\n✓ Identified {len(high_risk)} high-risk customers")

print("\nHigh-Risk Customers Sample:")
print(high_risk[['customer_id', 'tenure', 'contract_type', 'monthly_charges', 'churn_probability']].head(10))

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print("\n✓ Data successfully loaded from MySQL")
print("✓ Models trained and evaluated")
print("✓ Visualizations created")
print("✓ High-risk customers identified")