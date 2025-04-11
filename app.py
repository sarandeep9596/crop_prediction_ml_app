import numpy as np
from flask import Flask, request, render_template_string, send_from_directory
import pickle
import os

app = Flask(__name__, template_folder='.')

model = pickle.load(open("model.pkl", "rb"))

# Serve CSS
@app.route('/style.css')
def serve_css():
    return send_from_directory('.', 'style.css')

# Serve background image
@app.route('/<filename>')
def serve_image(filename):
    return send_from_directory('.', filename)

@app.route("/")
def Home():
    with open("index.html", "r") as f:
        html_content = f.read()
    return render_template_string(html_content, prediction_text="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)
        result = f"The Predicted Crop is {prediction[0]}"
    except Exception as e:
        result = f"Error: {str(e)}"

    with open("index.html", "r") as f:
        html_content = f.read()
    return render_template_string(html_content, prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
