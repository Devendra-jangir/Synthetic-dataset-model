from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('randomforest_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]

    prediction = model.predict(final_features)

    output = "Churn" if prediction == 1 else "Not Churn"

    return render_template('index.html', prediction_text=f'This customer is likely to: {output}')

if __name__ == "__main__":
    app.run(debug=True)
