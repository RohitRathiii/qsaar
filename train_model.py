from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from data_loader import load_data, preprocess_data
from ayurbot_dataset import create_data_loader
from sklearn.model_selection import train_test_split
import json

from transformers import DistilBertTokenizerFast

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


texts, labels, label_dict = preprocess_data(load_data())

with open('label_dict.json', 'w') as file:
    json.dump(label_dict, file)

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

train_loader = create_data_loader(train_texts, train_labels, tokenizer)
val_loader = create_data_loader(val_texts, val_labels, tokenizer)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_dict))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
)

trainer.train()

model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
