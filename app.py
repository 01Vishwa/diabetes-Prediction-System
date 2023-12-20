from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

data = pd.read_csv("E:\Diabetes_Prediction_Analysis-main\diabetes.csv")

x = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_scaled, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        values = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        values_scaled = scaler.transform([values])

        print("Form Data:", values)
        print("Scaled Input:", values_scaled)

        prediction = knn_model.predict(values_scaled)

        print("Prediction:", prediction)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
