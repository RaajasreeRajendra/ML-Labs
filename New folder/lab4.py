import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


# EXPERIMENT 1 - Simple Linear Regression

print("Experiment 1 - Linear Regression")

data = pd.read_csv("TvMarketing.csv")

print(data.head())

X = data[['TV']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

y_pred = model.predict(X_test)

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.title("Best Fit Line")
plt.show()

print("Actual vs Predicted")
print(pd.DataFrame({'Actual':y_test, 'Predicted':y_pred}))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)


# EXPERIMENT 2 - Multiple Linear Regression

print("\nExperiment 2 - Multiple Linear Regression")

df = pd.read_csv("co2.csv")

print(df.head())

sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

df[['Volume','Weight','CO2']].boxplot()
plt.title("Boxplot")
plt.show()

sns.scatterplot(x='Volume',y='CO2',data=df)
plt.show()

sns.scatterplot(x='Weight',y='CO2',data=df)
plt.show()

X = df[['Volume','Weight']]
y = df['CO2']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model2 = LinearRegression()
model2.fit(X_train,y_train)

print("Intercept:",model2.intercept_)
print("Weights:",model2.coef_)

y_pred = model2.predict(X_test)

print("MAE:",mean_absolute_error(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred)))

plt.plot(y_test.values,label="Actual")
plt.plot(y_pred,label="Predicted")
plt.legend()
plt.title("Actual vs Predicted CO2")
plt.show()


# EXPERIMENT 3 - Logistic Regression

print("\nExperiment 3 - Logistic Regression")

df = pd.read_csv("ad_click.csv")

print(df.info())

df = df.dropna()

numeric = df.select_dtypes(include=np.number)

sns.heatmap(numeric.corr(),annot=True)
plt.show()

X = numeric.drop('Clicked on Ad',axis=1)
y = numeric['Clicked on Ad']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model3 = LogisticRegression(max_iter=1000)
model3.fit(X_train,y_train)

y_pred = model3.predict(X_test)

print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.title("Confusion Matrix")
plt.show()

fpr,tpr,_ = roc_curve(y_test,model3.predict_proba(X_test)[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr,label="ROC curve")
plt.legend()
plt.show()

precision,recall,_ = precision_recall_curve(y_test,model3.predict_proba(X_test)[:,1])

plt.plot(recall,precision)
plt.title("Precision Recall Curve")
plt.show()

kfold = KFold(n_splits=5)
scores = cross_val_score(model3,X,y,cv=kfold)

print("K-Fold Scores:",scores)
print("Mean Accuracy:",scores.mean())
