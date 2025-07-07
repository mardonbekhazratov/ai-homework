import joblib

def save_model(model, filename="model.joblib"):
    joblib.dump(model, filename)
    
def load_model(filename="model.joblib"):
    loaded_model = joblib.load(filename)
    return loaded_model