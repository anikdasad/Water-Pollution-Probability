import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('water_potability.csv')

# Handle missing values
df = df.fillna(df.mean())

# Prepare features and target
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'Support Vector Machine': SVC(random_state=42, probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Results storage
results = []

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    
    # Train model
    if name in ['Support Vector Machine', 'K-Nearest Neighbors']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation score
    if name in ['Support Vector Machine', 'K-Nearest Neighbors']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV Mean': cv_mean,
        'CV Std': cv_std
    })

# Create comparison table
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.round(4)

# Display the comparison table
print("\n" + "="*80)
print("MACHINE LEARNING MODELS COMPARISON FOR WATER POTABILITY PREDICTION")
print("="*80)
print(comparison_df.to_string(index=False))

# Save to CSV
comparison_df.to_csv('model_comparison_results.csv', index=False)
print(f"\nResults saved to 'model_comparison_results.csv'")

# Create a formatted markdown table
markdown_table = comparison_df.to_markdown(index=False)
with open('model_comparison_table.md', 'w') as f:
    f.write("# Machine Learning Models Comparison for Water Potability Prediction\n\n")
    f.write(markdown_table)
    f.write("\n\n## Key Findings:\n")
    f.write("- **Best Overall Model**: Based on F1-Score and CV performance\n")
    f.write("- **Most Stable Model**: Based on CV Standard Deviation\n")
    f.write("- **Fastest Model**: Based on training time\n")

print(f"Markdown table saved to 'model_comparison_table.md'")

# Create visualization
plt.figure(figsize=(12, 8))

# Metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.1

for i, (name, _) in enumerate(models.items()):
    values = [comparison_df.loc[i, metric] for metric in metrics]
    plt.bar(x + i*width, values, width, label=name, alpha=0.8)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width*3, metrics)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('model_comparison_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Find best model
best_model_idx = comparison_df['F1-Score'].idxmax()
best_model = comparison_df.loc[best_model_idx, 'Model']

print(f"\n" + "="*50)
print(f"BEST PERFORMING MODEL: {best_model}")
print(f"F1-Score: {comparison_df.loc[best_model_idx, 'F1-Score']:.4f}")
print(f"Accuracy: {comparison_df.loc[best_model_idx, 'Accuracy']:.4f}")
print(f"Cross-Validation Score: {comparison_df.loc[best_model_idx, 'CV Mean']:.4f} (+/- {comparison_df.loc[best_model_idx, 'CV Std']:.4f})")
print("="*50) 