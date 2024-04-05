import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset into a pandas DataFrame
df = pd.read_csv("/Users/gayathriks/Desktop/Athena Pro/athenadata01.csv")

# Step 2: Remove commas and convert the numerical columns to float
numeric_cols = ["FootPressure1", "FootPressure2 ", "FootPressure3", "FootPressure4"]


# Step 3: Check for missing values
df.isnull().sum()

# Step 4: Display the first few rows of the data
df.head()

# Step 5: Generate a correlation heatmap
import seaborn as sns

sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="RdYlBu")
corr = df.corr()["Status"].sort_values(ascending=False).to_frame()
plt.figure(figsize=(9, 9))
sns.heatmap(corr, cmap="RdYlBu", annot=True)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.impute import SimpleImputer

# Step 6: Prepare the data for training
X = df[numeric_cols]
y = df["Status"]

# Drop rows with NaN values in y
df_clean = df.dropna(subset=["Status"])
X = df_clean[numeric_cols]
y = df_clean["Status"]

# Step 7: Impute missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Step 8: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# Step 9: Train the Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
tree_predict = clf.predict(X_test)
print(clf.predict([[36.7, 207, 96, 85, 48]]))
clf.score(X_test, y_test)
from sklearn.ensemble import RandomForestClassifier

rand_classifier = RandomForestClassifier()
rand_classifier.fit(X_train, y_train)
r = rand_classifier.predict(X_test)
rand_classifier.score(X_test, y_test)
from sklearn.svm import SVC

machine = SVC()
machine.fit(X_train, y_train)
m = machine.predict(X_test)
machine.predict([[40.1, 190, 86, 53, 87]])
machine.score(X_test, y_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
k = knn.predict(X_test)
knn.score(X_test, y_test)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
l = lr.predict(X_test)
lr.predict([[36.1, 160, 86, 53, 87]])
lr.score(X_test, y_test)
from sklearn.metrics import confusion_matrix

# For Random forest classifier
print(confusion_matrix(y_test, r))
sns.heatmap(confusion_matrix(y_test, r), annot=True)
# For decision tree Classifier
print(confusion_matrix(y_test, tree_predict))
sns.heatmap(confusion_matrix(y_test, tree_predict), annot=True)
# For KNN classifier
print(confusion_matrix(y_test, k))
sns.heatmap(confusion_matrix(y_test, k), annot=True)
# For SVM classifier
print(confusion_matrix(y_test, m))
sns.heatmap(confusion_matrix(y_test, m), annot=True)
# For Logistic regression
print(confusion_matrix(y_test, l))
sns.heatmap(confusion_matrix(y_test, l), annot=True)


def consult(A):
    if A[0] == 1:
        return "Consult doctor"
    if A[0] == 0:
        return "U r Fine"


A = clf.predict([[40.1, 190, 86, 53, 87]])
consult(A)
