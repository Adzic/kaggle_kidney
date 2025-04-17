import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Download latest version
path = kagglehub.dataset_download("amanik000/kidney-disease-dataset")
print("Path to dataset files:", path)

# Load the dataset
csv_path = f"{path}/kidney_disease_dataset.csv"
df = pd.read_csv(csv_path)

# EDA
print("Dataset shape:", df.shape)
print("First 5 rows:")
print(df.head())
print("\nInfo:")
df.info()
print("\nSummary statistics:")
print(df.describe(include='all'))
print("\nMissing values per column:")
print(df.isnull().sum())

# Visualize class distribution
plt.figure(figsize=(8,4))
df['Target'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Target')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
print("Class distribution plot saved as class_distribution.png")

# --- Feature Engineering ---
bins = [0, 18, 40, 60, 100]
labels = ['child', 'young_adult', 'adult', 'senior']
df['age_group'] = pd.cut(df['Age of the patient'], bins=bins, labels=labels, right=False)
if 'Serum creatinine (mg/dl)' in df.columns and 'Blood urea (mg/dl)' in df.columns:
    df['urea_creatinine_ratio'] = df['Blood urea (mg/dl)'] / (df['Serum creatinine (mg/dl)'] + 1e-5)
if 'Hemoglobin level (gms)' in df.columns and 'Packed cell volume (%)' in df.columns:
    df['hemoglobin_pc_ratio'] = df['Hemoglobin level (gms)'] / (df['Packed cell volume (%)'] + 1e-5)
low_var_cols = [col for col in df.columns if df[col].nunique() <= 1]
if low_var_cols:
    print("Dropping low-variance columns:", low_var_cols)
    df = df.drop(columns=low_var_cols)
corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
if to_drop:
    print("Dropping highly correlated columns:", to_drop)
    df = df.drop(columns=to_drop)

# Encode categorical variables
le_dict = {}
for col in df.select_dtypes(include=['object', 'category']).columns:
    if col != 'Target':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
# Encode target
target_le = LabelEncoder()
df['Target'] = target_le.fit_transform(df['Target'])

# Split features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- Data Balancing: SMOTEENN ---
smoteenn = SMOTEENN(random_state=42)
X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train, y_train)
print("After SMOTEENN, train class distribution:", pd.Series(y_train_smoteenn).value_counts().to_dict())

# --- Data Balancing: SMOTE only ---
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("After SMOTE, train class distribution:", pd.Series(y_train_smote).value_counts().to_dict())

# --- Data Balancing: Class weights ---
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

# --- Model Definitions ---
xgb = XGBClassifier(tree_method='hist', eval_metric='mlogloss', random_state=42, n_estimators=200, max_depth=6, learning_rate=0.1)
rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42, early_stopping=True)

# --- 1. Stacking Ensemble (SMOTEENN) ---
stack = StackingClassifier(
    estimators=[('xgb', xgb), ('rf', rf), ('mlp', mlp)],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    cv=3, n_jobs=-1, passthrough=True
)
stack.fit(X_train_smoteenn, y_train_smoteenn)
y_pred_stack = stack.predict(X_test)
print("\nStacking Ensemble (SMOTEENN) Classification Report:")
print(classification_report(y_test, y_pred_stack, target_names=target_le.classes_))
print("Stacking Ensemble Accuracy:", accuracy_score(y_test, y_pred_stack))

# --- 2. Voting Ensemble (SMOTE) ---
voting = VotingClassifier(
    estimators=[('xgb', xgb), ('rf', rf), ('mlp', mlp)],
    voting='soft', n_jobs=-1
)
voting.fit(X_train_smote, y_train_smote)
y_pred_vote = voting.predict(X_test)
print("\nVoting Ensemble (SMOTE) Classification Report:")
print(classification_report(y_test, y_pred_vote, target_names=target_le.classes_))
print("Voting Ensemble Accuracy:", accuracy_score(y_test, y_pred_vote))

# --- 3. XGBoost (class weights) ---
xgb_weighted = XGBClassifier(tree_method='hist', eval_metric='mlogloss', random_state=42, n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=None)
xgb_weighted.fit(X_train, y_train, sample_weight=np.array([class_weight_dict[c] for c in y_train]))
y_pred_xgb_weighted = xgb_weighted.predict(X_test)
print("\nXGBoost (class weights) Classification Report:")
print(classification_report(y_test, y_pred_xgb_weighted, target_names=target_le.classes_))
print("XGBoost (class weights) Accuracy:", accuracy_score(y_test, y_pred_xgb_weighted))

# --- 4. Random Forest (class weights) ---
rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200)
rf_weighted.fit(X_train, y_train)
y_pred_rf_weighted = rf_weighted.predict(X_test)
print("\nRandom Forest (class weights) Classification Report:")
print(classification_report(y_test, y_pred_rf_weighted, target_names=target_le.classes_))
print("Random Forest (class weights) Accuracy:", accuracy_score(y_test, y_pred_rf_weighted))

# --- 5. Anomaly Detection (Isolation Forest) ---
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train)
anomaly_scores = iso_forest.decision_function(X_test)
anomaly_pred = iso_forest.predict(X_test)
# -1 means anomaly, 1 means normal
anomaly_flagged = (anomaly_pred == -1)
print(f"\nIsolation Forest flagged {anomaly_flagged.sum()} anomalies out of {len(anomaly_flagged)} test samples.")

# --- Confusion Matrices ---
cm_stack = confusion_matrix(y_test, y_pred_stack)
cm_vote = confusion_matrix(y_test, y_pred_vote)
cm_xgb_weighted = confusion_matrix(y_test, y_pred_xgb_weighted)
cm_rf_weighted = confusion_matrix(y_test, y_pred_rf_weighted)
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Blues', xticklabels=target_le.classes_, yticklabels=target_le.classes_)
plt.title('Stacking (SMOTEENN)')
plt.subplot(2,2,2)
sns.heatmap(cm_vote, annot=True, fmt='d', cmap='Greens', xticklabels=target_le.classes_, yticklabels=target_le.classes_)
plt.title('Voting (SMOTE)')
plt.subplot(2,2,3)
sns.heatmap(cm_xgb_weighted, annot=True, fmt='d', cmap='Purples', xticklabels=target_le.classes_, yticklabels=target_le.classes_)
plt.title('XGBoost (class weights)')
plt.subplot(2,2,4)
sns.heatmap(cm_rf_weighted, annot=True, fmt='d', cmap='Oranges', xticklabels=target_le.classes_, yticklabels=target_le.classes_)
plt.title('Random Forest (class weights)')
plt.tight_layout()
plt.savefig('all_confusion_matrices.png')
print("All confusion matrices saved as all_confusion_matrices.png")

# --- Feature Importances ---
xgb_single = XGBClassifier(tree_method='hist', eval_metric='mlogloss', random_state=42, n_estimators=200, max_depth=6, learning_rate=0.1)
xgb_single.fit(X_train_smote, y_train_smote)
plt.figure(figsize=(10,5))
importances = xgb_single.feature_importances_
indices = importances.argsort()[::-1][:15]
plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=90)
plt.title('Top 15 XGBoost Feature Importances (SMOTE)')
plt.tight_layout()
plt.savefig('xgb_feature_importance_fixed.png')
print("XGBoost feature importance plot saved as xgb_feature_importance_fixed.png")

rf_single = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200)
rf_single.fit(X_train_smote, y_train_smote)
plt.figure(figsize=(10,5))
rf_importances = rf_single.feature_importances_
indices_rf = rf_importances.argsort()[::-1][:15]
plt.bar(range(len(indices_rf)), rf_importances[indices_rf], align='center', color='g')
plt.xticks(range(len(indices_rf)), [X.columns[i] for i in indices_rf], rotation=90)
plt.title('Top 15 Random Forest Feature Importances (SMOTE)')
plt.tight_layout()
plt.savefig('rf_feature_importance_fixed.png')
print("Random Forest feature importance plot saved as rf_feature_importance_fixed.png")
