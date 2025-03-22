from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load BERT pre-trained model
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Tenant review
review = "The maintenance service was quick and helpful."

inputs = tokenizer.encode_plus(review, return_tensors='pt')
output = model(**inputs)
sentiment = torch.argmax(output.logits)  # Positive/Negative label
print(f"Sentiment Score: {sentiment}")
