import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset shape:", X.shape)

# Train test split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)

# -----------------------------
# Experiment 1
# Naive Bayes

print("\nNaive Bayes Model")

nb = GaussianNB()
nb.fit(X_train,y_train)

y_pred_nb = nb.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred_nb))

print("\nClassification Report")
print(classification_report(y_test,y_pred_nb))

cm_nb = confusion_matrix(y_test,y_pred_nb)

sns.heatmap(cm_nb,annot=True)
plt.title("Naive Bayes Confusion Matrix")
plt.show()

precision,recall,_ = precision_recall_curve(y_test,nb.predict_proba(X_test)[:,1])
plt.plot(recall,precision)
plt.title("Precision Recall Curve - Naive Bayes")
plt.show()

# -----------------------------
# Experiment 2
# Decision Tree

print("\nDecision Tree Model")

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

y_pred_dt = dt.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred_dt))

print("\nClassification Report")
print(classification_report(y_test,y_pred_dt))

cm_dt = confusion_matrix(y_test,y_pred_dt)

sns.heatmap(cm_dt,annot=True)
plt.title("Decision Tree Confusion Matrix")
plt.show()

plt.figure(figsize=(10,6))
plot_tree(dt,max_depth=3,filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# -----------------------------
# Experiment 3
# Comparison

train_acc_nb = nb.score(X_train,y_train)
test_acc_nb = nb.score(X_test,y_test)

train_acc_dt = dt.score(X_train,y_train)
test_acc_dt = dt.score(X_test,y_test)

models = ["Naive Bayes","Decision Tree"]
train_scores = [train_acc_nb,train_acc_dt]
test_scores = [test_acc_nb,test_acc_dt]

x = np.arange(len(models))

plt.bar(x-0.2,train_scores,0.4,label="Train Accuracy")
plt.bar(x+0.2,test_scores,0.4,label="Test Accuracy")
plt.xticks(x,models)
plt.title("Model Accuracy Comparison")
plt.legend()
plt.show()

# ROC comparison
fpr_nb,tpr_nb,_ = roc_curve(y_test,nb.predict_proba(X_test)[:,1])
fpr_dt,tpr_dt,_ = roc_curve(y_test,dt.predict_proba(X_test)[:,1])

plt.plot(fpr_nb,tpr_nb,label="Naive Bayes")
plt.plot(fpr_dt,tpr_dt,label="Decision Tree")

plt.title("ROC Curve Comparison")
plt.legend()
plt.show()