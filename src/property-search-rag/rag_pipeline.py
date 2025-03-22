from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Step 1: Embed property docs
property_data = ["2BHK apartment in sector 45", "Studio apartment near mall"]
embedding = OpenAIEmbeddings()
db = FAISS.from_texts(property_data, embedding)

# Step 2: Set up RAG pipeline
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

# Step 3: Query
query = "Is there a studio apartment available?"
result = qa_chain.run(query)
print(result)
