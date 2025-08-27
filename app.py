import os
import json
import joblib
import  numpy as np 
from flask import Flask, request, jsonify

#__config__
MODEL_PATH = os.getenv('MODEL_PATH', "model/iris_model.pkl")

app = Flask(__name__)
# load once at startup
try:
  model = joblib.load(MODEL_PATH)
except Exception as e:
  raise RuntimeError(f"could not load model from {MODEL_PATH}: {e}")
@app.get("/health")
def health():
  return {"status": "ok"}, 200
@app.post("/predict")
def predict():
  try:
    payload = request.get_json(force=True)
    x = payload.get('input')
    if x is None:
      return jsonify(error="Missing input"), 400
    # Normalize 2d array
    if isinstance(x, list) and (len(x) == 0 or not isinstance(x[0], list)):
      x = [x]
    x = np.array(x, dtype=float)
    preds = model.predict(x)
    preds = preds.tolist()
    return jsonify(predictions=preds), 200
  except Exception as e:
    return jsonify(error=str(e)), 500
if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

