from flask import Flask, render_template, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import json

app = Flask(__name__)

# Load label_dict
with open('label_dict.json', 'r') as file:
    label_dict = json.load(file)

tokenizer = DistilBertTokenizer.from_pretrained('./model')
model = DistilBertForSequenceClassification.from_pretrained('./model')

nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

@app.route('/')
def index():
    # Renders index.html as the main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['message']
    predictions = nlp(text)

    # 'label' returned by the pipeline is a string like 'LABEL_0'
    # Extract the numeric part and convert to an integer
    label_id = int(predictions[0]['label'].split('_')[-1])
    
    # Make sure label_id is in label_dict
    if label_id in label_dict:
        response_tag = label_dict[label_id]
        response = select_response_for_tag(response_tag)  # Make sure this function is defined
    else:
        response = "Sorry, I am not sure how to respond to that."

    return jsonify(response=response)

if __name__ == '__main__':
    app.run(debug=True)
