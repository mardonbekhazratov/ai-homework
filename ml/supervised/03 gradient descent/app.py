import numpy as np
import joblib
from pathlib import Path

current_dir = Path(__file__).resolve().parent

tv = 230.1 # float(input('TV: '))
radio = 37.8 # float(input('Radio: '))
newspaper = 69.2 # float(input('Newspaper: '))

sclr = joblib.load(current_dir / 'scaler.joblib')
model = joblib.load(current_dir / 'lin_reg.joblib')

x = np.array([[tv, radio, newspaper]])

# Transform
x = sclr.transform(x)
# x = (x - np.array([138.679375,  23.94375 ,  30.308125])) / np.array([84.40288139, 15.07727051, 20.81144809])

# # Predict
# pred = x @ np.array([ 4.75338342,  1.60461246, -0.09364872]) + np.float64(14.826875)
pred = model.predict(x)

print(f"Prediction is {pred.item():.2f}")


