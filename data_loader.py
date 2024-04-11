import json
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def load_data(filename='data.json'):
    with open(filename) as file:
        data = json.load(file)
    return data['intents']

def preprocess_data(intents):
    texts, labels = [], []
    label_dict = {}
    for idx, intent in enumerate(intents):
        for pattern in intent['patterns']:
            texts.append(pattern)
            labels.append(idx)
        label_dict[idx] = intent['tag']
    return texts, labels, label_dict

if __name__ == "__main__":
    intents = load_data()
    texts, labels, label_dict = preprocess_data(intents)
    print(f"Loaded {len(texts)} texts.")
