from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------
# Vector DB
# ---------------------------------------
PERSIST_DIRECTORY = "./chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model
)

# ---------------------------------------
# Retriever
# ---------------------------------------
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# ---------------------------------------
# LLM (FREE - Local)
# ---------------------------------------
llm = ChatOllama(
    model="llama3:latest",      # or qwen3:8b
    temperature=0
)

# ---------------------------------------
# Prompt Template
# ---------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not found in the context, say:
"I could not find that information in the documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ---------------------------------------
# Main Loop
# ---------------------------------------
while True:

    query = input("\nAsk a question: ").strip()

    if query.lower() in ["exit", "quit"]:
        print("\nGoodbye!")
        break

    print("\nSearching documents...")

    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        print("\nNo relevant documents found.")
        continue

    # Combine retrieved chunks
    context = "\n\n".join(
        [doc.page_content for doc in relevant_docs]
    )

    # print("\nRetrieved Chunks:")
    # print("-" * 50)

    # for i, doc in enumerate(relevant_docs, start=1):
    #     print(f"\nChunk {i}")
    #     print(doc.page_content[:200])
    #     print("-" * 50)

    # Create chain
    chain = prompt | llm

    # Generate answer
    response = chain.invoke({
        "context": context,
        "question": query
    })

    print("\nAnswer:")
    print(response.content)