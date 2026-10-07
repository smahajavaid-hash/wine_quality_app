import pandas as pd 
df=pd.read_csv("winequality-red.csv")
print(df)
X=df.drop("quality",axis=1)
Y=df["quality"]
print(X)
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2 , random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
model=LinearRegression()
model.fit(X_train,Y_train)
line_pred=model.predict(X_test)
mae=mean_absolute_error(Y_test,line_pred)
r2=r2_score(Y_test,line_pred)
# print(mae)
# print(r2)
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(
     n_estimators=100,
     random_state=42
)
rf_model.fit(X_train ,Y_train)
rf_pred=rf_model.predict(X_test)
rf_mae= mean_absolute_error(Y_test,rf_pred)
rf_r2=r2_score(Y_test,rf_pred)
print(rf_mae)
print(rf_r2)
print("rf")
import joblib
joblib.dump(rf_model, 
r"C:\Users\smaha\OneDrive\Desktop\vine quality\wine_quality_score.pkl")
model_rf=joblib.load("wine_quality_score.pkl")
model_pred=model_rf.predict(X_test)
print(model_pred)