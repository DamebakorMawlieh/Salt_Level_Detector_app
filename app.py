from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model using joblib
model = joblib.load('model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    voltage = data.get('voltage')

    # Input validation
    if voltage is None or not isinstance(voltage, (int, float)):
        return jsonify({'error': 'Invalid input. Voltage must be a number.'}), 400

    # Make prediction
    prediction = model.predict([[voltage]])

    # Return the prediction as a JSON response
    return jsonify({'salt_level': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)