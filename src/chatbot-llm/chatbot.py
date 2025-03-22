from transformers import pipeline

# Load pre-trained LLM model
chatbot = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Sample interaction
tenant_query = "What are the available apartments in sector 45?"
response = chatbot(tenant_query, max_length=50)
print(response[0]['generated_text'])
