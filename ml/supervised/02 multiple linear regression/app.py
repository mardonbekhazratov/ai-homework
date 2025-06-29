import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = input("Enter TV,Radio,Newspaper: ")
x = x.split(',')
x = [float(i.strip()) for i in x]
x = np.array([x])

# For Standard Scaler
mean_ = np.array([156.38928571,  23.525     ,  30.44785714])
scale_ = np.array([83.59965712, 14.58321503, 20.62458473])

# For model weights
coef_ = np.array([ 4.48646421,  1.52054778,  0.12062109,  0.51359714, -0.11101447,
        0.11551558])
intercept_ = np.float64(15.71249489788946)

# Apply transformation
x1 = (x - mean_) / scale_
x1 = np.c_[x1, x1[:,0]*x1[:,1], x1[:,0]*x1[:,2], x1[:,1]*x1[:,2]]

# print(x1)
y_pred1 = x1 @ coef_ + intercept_
y_pred1 = y_pred1[0]



dirname = os.path.dirname(__file__)
std_scl: StandardScaler = joblib.load(os.path.join(dirname, 'standard_scaler.joblib'))
make_poly: PolynomialFeatures = joblib.load(os.path.join(dirname, 'make_poly.joblib'))
lin_reg: LinearRegression = joblib.load(os.path.join(dirname, 'linear_regression.joblib'))

x = std_scl.transform(x)
x = make_poly.transform(x)

y_pred = lin_reg.predict(x)
y_pred = y_pred[0]

print(f"Expected Sales is {y_pred:.3f}")
print(f"Expected Sales is {y_pred1:.3f}")

# Streamlit
# Gradio

