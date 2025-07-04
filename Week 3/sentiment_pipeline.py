from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load the IMDb dataset
dataset = load_dataset("imdb")
# Load tokenizer for bert-base-uncased
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define a tokenization function
def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=512)

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# Load BERT with a classification head (2 labels for positive/negative)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# Rename columns and format for PyTorch
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

# Metrics function
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)
trainer.train()
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
# Save model and tokenizer
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

# For inference: Reload model
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="./sentiment_model", tokenizer="./sentiment_model")

# Test on custom input
result = sentiment_pipeline("The movie was absolutely fantastic!")
print(result)
