import pickle
import os
print("\n")
print(os.getcwd())

dv_file = "dv.bin"

with open(dv_file,"rb") as file:
    dv = pickle.load(file)

model_file = "model1.bin"

with open(model_file,"rb") as file:
    model = pickle.load(file)

customer = {
    "job": "management", 
    "duration": 400, 
    "poutcome": "success"
}

X = dv.transform([customer])
print(model.predict_proba(X)[0,1])