import pandas as pd
data=pd.read_csv("earthquake dataset.csv")
print("shape of dataset:")
print(data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nColumn names:")
print(data.columns)
data.head()
data.columns
data.info()
data.columns=data.columns.str.strip()
data=data.drop(columns=['nst'])
print(data.columns)
data=data.fillna(data.mean(numeric_only=True))
print(data.isnull().sum())
data=data.drop(columns=['id','place','status','updated','locationSource','magSource','data_type','net'],errors='ignore')
print(data.head())
print(data.isnull().sum())
data=data[['latitude','longitude','depth','mag','gap','dmin','rms']]
print(data.head())
x=data.drop('mag',axis=1)
y=data['mag']
print(x.head())
print(y.head())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
print("Model Training Completed")
prediction=model.predict(x_test)
print(prediction[:5])
y_pred=model.predict(x_test)
print(y_pred[:5])
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
from sklearn.metrics import mean_absolute_error,r2_score
print("MAE:",mean_absolute_error(y_test,prediction))
print("R2 Score:",r2_score(y_test,prediction))
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.scatter(y_test,prediction)
plt.xlabel("Actual Magnitude")
plt.ylabel("Predicted Magnitude")
plt.title("Actual vs Predicted Earthquake Magnitude")
plt.subplot(1,3,2)
plt.plot(y_test.values[:50],label='Actual')
plt.plot(prediction[:50],label='Predicted')
plt.title("Prediction Line")
plt.legend()
plt.subplot(1,3,3)
sns.heatmap(data.corr(),annot=True,cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()











