from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load pickled objects
with open("models/cv.pkl", "rb") as f:
    cv = pickle.load(f)

with open("models/clf.pkl", "rb") as f:
    clf = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html", prediction=None, email="")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('content')
    tokenized_email = cv.transform([email])  # Transform input email
    prediction = clf.predict(tokenized_email)[0]  # Make prediction
    prediction_label = "Spam" if prediction == 1 else "Not Spam"
    return render_template("index.html", prediction=prediction_label, email=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
