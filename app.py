from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("house_price_model.pkl")  # Make sure this file is in the same folder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['area']),
        int(request.form['bedrooms']),
        int(request.form['bathrooms']),
        int(request.form['parking']),
        int(request.form['floors']),
        int(request.form['age']),
        int(request.form['balconies']),
        int(request.form['near_metro']),
        float(request.form['school_distance']),
        int(request.form['city'])
    ]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction=f"₹{prediction:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)
