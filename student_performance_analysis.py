# 📊 Intern Intelligence - Task 2: Data Analysis & Modeling
# Dataset: Student Performance Prediction
# Author: Maryam Yasha

# ========== IMPORTING LIBRARIES ==========
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns  #for drawing charts
import matplotlib.pyplot as plt #for showing charts
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# ========== STEP 1: LOAD DATA ==========
df = pd.read_csv("StudentsPerformance.csv") #reads the CSV file and store in df
# print(df.info())
# print(df.isnull().sum())

# ========== STEP 2: FEATURE ENGINEERING ==========
#create new column average score from math, reading, writing
df['average'] = (df['math score'] + df['reading score'] + df['writing score'])/3

#create new colunm result
df['result'] = df['average'].apply(lambda x: 1 if x >=40 else 0) # 1=pass 0=fail

print("\n🔹 Sample Data with Average & Result:")
print(df[['math score', 'reading score', 'writing score', 'average', 'result']].head())

# For better readability, convert result to text (pass/fail)
df['result'] = df['result'].map({1: 'pass', 0: 'fail'})
print("\nPass/Fail Count:")
print(df['result'].value_counts())

# ========== STEP 3: LABEL ENCODING ==========
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object': # means text
        df[col] = le.fit_transform(df[col])

print("\n🔹 Data after Label Encoding:")
print(df.head())

# ========== STEP 4: EDA (Exploratory Data Analysis) ==========
# Bar chart: Overall pass/fail
df_plot = df.copy()
df_plot['result'] = df_plot['result'].map({1: 'pass', 0: 'fail'})
plt.figure(figsize=(6,4))
sns.countplot(x='result', hue='result', data=df_plot, palette='Set2', legend=False)
plt.title("Pass vs Fail")
plt.savefig("pass_vs_fail_chart.png")
plt.show()

# Bar chart: Gender vs Pass/Fail
df_plot = df.copy()
df_plot['gender'] = df_plot['gender'].map({0: 'male', 1: 'female'})
df_plot['result'] = df_plot['result'].map({1: 'pass', 0: 'fail'})
plt.figure(figsize=(6,4))
sns.countplot(x='gender', hue='result', data=df_plot, palette='Set2')
plt.title("Gender vs Pass/Fail")
plt.savefig("gender_vs_pass_fail_chart.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()

sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f",  # Show 2 decimal places
    annot_kws={"size": 10},  # Larger annotation text
    linewidths=0.5,
    vmin=-1, vmax=1  # Fixed color scale
)

# Rotate x-axis labels and make them readable
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title("Feature Correlation Heatmap", pad=20)  # Add padding
plt.tight_layout()  # Prevent label cutoff
plt.savefig("correlation_heatmap.png")
plt.show()

# ========== STEP 5: MODEL BUILDING ==========
# Split data into input (X) and output (y)
X = df.drop(['result', 'average'], axis=1)
y = df['result']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ========== STEP 6: EVALUATION ==========
accuracy = accuracy_score(y_test, y_pred)

# Save evaluation results to file
fo = open('Classification_Report.txt', 'w')
fo.write('MODEL PERFORMANCE METRICS\n\n')
fo.write("Summary of Model Evaluation:\n")
fo.write(f"Accuracy: {round(accuracy * 100, 2)}% \n\n")

fo.write("Confusion Matrix:\n")
fo.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")

fo.write("Detailed Classification Report:\n")
fo.write(classification_report(y_test, y_pred))
fo.close()
print("\n✅ Model training and evaluation Saved!")

# ========== STEP 7: HYPERPARAMETER OPTIMIZATION ==========
param_grid ={
    'n_estimators': [50, 100, 150],
    'max_depth':[None, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train) #.fit() tests all combinations and selects the best one

print("Best Parameters:", grid.best_params_)
print("Best Accuracy (CV):", round(grid.best_score_ * 100, 2), "%")

fo = open("Classification_Report.txt", "a")
fo.write("\nHYPERPARAMETER OPTIMIZATION\n\n")
fo.write("Best Parameters: " + str(grid.best_params_) + "\n")
fo.write("Best Cross-Validation Accuracy: " + str(round(grid.best_score_ * 100, 2)) + "%\n\n")
fo.close()

print("\nClassification report saved as 'Classification_Report.txt'")
with open("Classification_Report.txt", "r") as f:
    print("\nReport Preview:\n")
    print(f.read())
