import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics 
import  accuracy_score,recall_score,f1_score,precision_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r"creditcard.csv")

print("shape : \n",df.shape)

print("\nClass Distribution:")
print(df['Class'].value_counts())

X = df.drop("Class",axis=1)
y = df["Class"]

print("X shape ",X.shape,"\ny shape ",y.shape)


""" Train test split """

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("X train ",X_train.shape,"\nX test ",X_test.shape)
print("y train ",y_train.shape,"\ny test ",y_test.shape)

""" Scaling """

scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

print("X train Scale ",X_train_scaled.shape,"\nX test Scale ",X_test_scaled.shape)

""" Model training before balancing """

'''model = LogisticRegression()
model.fit(X_train_scaled,y_train)
prediction = model.predict(X_test_scaled)

print("Accuracy : ",accuracy_score(y_test,prediction))
print("Precision : ",precision_score(y_test,prediction))
print("Recall : ",recall_score(y_test,prediction))
print("F1_score : ",f1_score(y_test,prediction))'''

""" Balancing """

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

"""print("Before SMOTE:", X_train_scaled.shape, y_train.shape)
print("After SMOTE:", X_res.shape, y_res.shape)"""


""" Model training after balancing """

model_2 = LogisticRegression(max_iter=1000,random_state=42)
model_2.fit(X_res,y_res)

y_pred = model_2.predict(X_test_scaled)

print("Logistic regression Accuracy :", accuracy_score(y_test, y_pred))
print("Logistic regression Precision :", precision_score(y_test, y_pred))
print("Logistic regression Recall :", recall_score(y_test, y_pred))
print("Logistic regression F1 Score :", f1_score(y_test, y_pred))

""" Model Evaluation """

#print("Confusion Matrix \n",confusion_matrix(y_test,y_pred))
#print("Classification report \n",classification_report(y_test,y_pred))

""" Visualization """

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = model_2.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)


auc_score = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc_score)

# Plot ROC curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0,1], [0,1], 'k--')  # diagonal baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

""" model for comparison """

tree = DecisionTreeClassifier()
tree.fit(X_res,y_res)
tree_predict = tree.predict(X_test_scaled)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [accuracy_score(y_test, y_pred), accuracy_score(y_test, tree_predict)],
    'Precision': [precision_score(y_test, y_pred), precision_score(y_test, tree_predict)],
    'Recall': [recall_score(y_test, y_pred), recall_score(y_test, tree_predict)],
    'F1': [f1_score(y_test, y_pred), f1_score(y_test, tree_predict)]
})
print(results)

import joblib

# Save your trained logistic regression model
joblib.dump(model_2, "fraud_model.pkl")

# Save your fitted scaler (important for deployment)
joblib.dump(scale, "scaler.pkl")

print(" Model and scaler saved successfully.")
