from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model

with open("random-forest-model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Employee Churn Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON input
        data = request.json  

        # Convert to DataFrame (Flask receives data as JSON, but ML models need DataFrame)
        df = pd.DataFrame([data])

        # Make a prediction
        prediction = model.predict(df)[0]  
        probability = model.predict_proba(df)[0][1]  

        # Return the response as JSON
        return jsonify({
            "Churn Prediction": int(prediction),  
            "Probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
