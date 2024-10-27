import pickle

from flask import Flask, request, jsonify

app = Flask("score")

dv_file = "dv.bin"

with open(dv_file,"rb") as file:
    dv = pickle.load(file)

model_file = "model1.bin"

with open(model_file,"rb") as file:
    model = pickle.load(file)

@app.route("/predict", methods = ["POST"])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    result ={
        "score_proba": float(y_pred)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

