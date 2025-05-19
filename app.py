from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'subject' not in data or 'content' not in data:
        return jsonify({'error': 'Missing subject or content'}), 400

    text = data['subject'] + ' ' + data['content']
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    return jsonify({'is_phishing': bool(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
