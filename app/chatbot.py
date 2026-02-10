from langchain.chains import RetrievalQA
from retriever import get_retriever
from llm import get_llm

retriever = get_retriever()
llm = get_llm()

rag = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

print("🤖 Enterprise RAG Chatbot (FREE)")
print("Type 'exit' to quit\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    answer = rag.run(query)
    print("Bot:", answer)
