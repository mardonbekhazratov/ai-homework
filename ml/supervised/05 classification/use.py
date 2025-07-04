import os
import pickle
from sklearn.linear_model import LogisticRegression

cur_dir = os.path.dirname(__file__)
file_path = os.path.join(cur_dir, 'log_reg.pkl')

with open(file_path, 'rb') as f:
    model: LogisticRegression = pickle.load(f)


model.predict()

