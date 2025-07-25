import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Data loaded. Shape:", df.shape)
print(df.head())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing_total = df['TotalCharges'].isna().sum()
print(f"Missing TotalCharges: {missing_total}")
df = df.dropna(subset=['TotalCharges'])
df = df.drop('customerID', axis=1)

for col in df.select_dtypes(include=['object']).columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

X = df.drop('Churn', axis=1)
y = df['Churn']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(4,4))
df['Churn'].value_counts().plot.pie(
    autopct='%1.1f%%', labels=['No', 'Yes'], colors=['#66b3ff','#ff9999']
)
plt.title('Churn Distribution')
plt.ylabel('')
plt.show()

plt.figure(figsize=(12,10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10,6))
plt.title("Feature Importances (Random Forest)")
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# --- Retention Action Function ---
def recommend_retention_actions(customer, feature_names, importances):
    actions = []
    important_features = [feature_names[i] for i in np.argsort(importances)[::-1][:5]]
    if customer['tenure'] < 6:
        actions.append("Offer a loyalty discount for new customers.")
    if customer['MonthlyCharges'] > np.percentile(df['MonthlyCharges'], 75):
        actions.append("Provide a personalized discount or bundle to reduce monthly charges.")
    if customer['Contract'] == 0:
        actions.append("Encourage switching to a longer-term contract with incentives.")
    if customer['TechSupport'] == 0:
        actions.append("Offer free or discounted tech support for 3 months.")
    if customer['OnlineSecurity'] == 0:
        actions.append("Provide a free trial of online security services.")
    if not actions:
        actions.append("Send a personalized thank you and ask for feedback.")
    return actions

# --- Predict churn for new data and suggest retention actions ---
new_customer = {
    'gender': 0,
    'SeniorCitizen': 0,
    'Partner': 1,
    'Dependents': 0,
    'tenure': 2,
    'PhoneService': 1,
    'MultipleLines': 0,
    'InternetService': 1,
    'OnlineSecurity': 0,
    'OnlineBackup': 1,
    'DeviceProtection': 1,
    'TechSupport': 0,
    'StreamingTV': 1,
    'StreamingMovies': 0,
    'Contract': 0,
    'PaperlessBilling': 1,
    'PaymentMethod': 2,
    'MonthlyCharges': 90.0,
    'TotalCharges': 180.0
}
new_df = pd.DataFrame([new_customer])
new_df_scaled = scaler.transform(new_df)
churn_pred = clf.predict(new_df_scaled)[0]
if churn_pred == 1:
    print("This customer is likely to leave (churn).")
    actions = recommend_retention_actions(new_customer, feature_names, importances)
    print("Recommended retention actions:")
    for act in actions:
        print("-", act)
else:
    print("This customer is likely to stay.")

