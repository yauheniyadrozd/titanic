import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Read the CSV file
df = pd.read_csv("titanic.csv")

# Data Exploration (commented out but useful for debugging)
# print(df.head())
# print(df.describe())
# print(df.info())
# print(df.isnull().sum())
# print(df['Age'].describe())

#================================================================================
# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())  # Using median due to skewed distribution
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill Embarked with most common value

# Feature Engineering
# One-Hot Encoding for Embarked
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')

# Create FamilySize feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Create Age Groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 16, 32, 48, 100],
                        labels=['child', 'young', 'adult', 'old'])
age_dummies = pd.get_dummies(df['AgeGroup'], prefix='Age')

# Encode Sex (male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Prepare feature matrix
X = pd.concat([df[['Pclass', 'Sex', 'Age', 'FamilySize']],
               embarked_dummies, age_dummies], axis=1)
y = df['Survived']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Feature Importance Analysis
print("\nFeature Importances:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(f"{name}: {importance:.3f}")

# Model Accuracy
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.4f}")

# =======================================================================

# 1. Feature Importance Plot
plt.figure(figsize=(10, 6))
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Most Important Features')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# 2. Age Distribution by Survival
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True,
             palette={0: 'red', 1: 'green'}, alpha=0.6)
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# 3. Survival Rate by Passenger Class
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=df, palette='viridis')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = pd.concat([X, y], axis=1).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

