import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
accuracy_score, confusion_matrix, roc_curve, roc_auc_score,
precision_recall_curve, average_precision_score, f1_score,
classification_report, matthews_corrcoef, precision_score,
recall_score, cohen_kappa_score, balanced_accuracy_score,
log_loss
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
file_path = r"C:\Users\Admin\OneDrive\Desktop\AD2\AD2 DRIVE\AD2 project"
df = pd.read_excel(file_path)
# Drop unnecessary columns (modify this based on your needs)
columns_to_drop = ['Last_name', 'Email', 'dob(DD-MM-YYYY)', 'Country_Code', 'Phone', 'Roll_no']
df = df.drop(columns_to_drop, axis=1)
# Handle missing values (you might want to customize this based on your dataset)
df = df.fillna(0)
# Convert 'Gender' and 'interested_for_placement' columns to strings
df['Gender'] = df['Gender'].astype(str)
df['interested_for_placement'] = df['interested_for_placement'].astype(str)
# Exclude non-numeric columns from features
X = df.select_dtypes(exclude=['object', 'datetime64'])
# Convert categorical columns to numerical using Label Encoding
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['interested_for_placement'] = label_encoder.fit_transform(df['interested_for_placement'])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['interested_for_placement'], test_size=0.2,
random_state=42)
random_forest_model = RandomForestClassifier()
logistic_regression_model = LogisticRegression()
knn_model = KNeighborsClassifier()
# List of classifiers for iteration
classifiers = [
('Random Forest', random_forest_model),
('Logistic Regression', logistic_regression_model),
('K-Nearest Neighbors', knn_model)
]
# Iterate through classifiers
for clf_name, clf in classifiers:
print(f'\n{clf_name} Classifier')
# Train the model
clf.fit(X_train, y_train)
# Make predictions on the test set
y_pred = clf.predict(X_test)
# Evaluate the model (excluding specific metrics for Logistic Regression and K-Nearest Neighbors)
if clf_name not in ['Logistic Regression', 'K-Nearest Neighbors']:
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
matthews_corr = matthews_corrcoef(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = confusion_matrix(y_test, y_pred)[0, 0] / (confusion_matrix(y_test, y_pred)[0, 0] +
confusion_matrix(y_test, y_pred)[0, 1])
cohen_kappa = cohen_kappa_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, clf.predict_proba(X_test))
print(f'Model Evaluation:')
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Matthews Correlation Coefficient: {matthews_corr:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'Cohen\'s Kappa: {cohen_kappa:.4f}')
print(f'Balanced Accuracy: {balanced_acc:.4f}')
print(f'Log Loss: {logloss:.4f}')
# Classification Report
class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 6))
plt.subplot(2, 4, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted') 
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# ROC Curve and AUC Score
y_probs = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)
plt.subplot(2, 4, 2)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# ROC Curve and AUC Score
y_probs = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)
plt.subplot(2, 4, 2)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
average_precision = average_precision_score(y_test, y_probs)
plt.subplot(2, 4, 3)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (Avg Precision = {average_precision:.2f})')
# Cross-Validation Scores Boxplot
cv_scores = cross_val_score(clf, X, df['interested_for_placement'], cv=5, scoring='accuracy')
plt.subplot(2, 4, 4)
sns.boxplot(cv_scores)
plt.title('Cross-Validation Scores')
plt.xlabel('Accuracy')
# Feature Importance (if applicable)
if hasattr(clf, 'feature_importances_'):
feature_importances = clf.feature_importances_
features = X.columns
plt.subplot(2, 4, 5)
sns.barplot(x=feature_importances, y=features, orient='h', palette='viridis')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title(f'{clf_name} Feature Importance')
plt.tight_layout()
plt.show()                                                     