from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Vector DB Location
persistent_directory = "./chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


while True:
    query = input("\nAsk a question: ").strip()

    # retriever = db.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={
    #         "k": 5,
    #         "score_threshold": 0.3
    #     }
    # )

    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

    if query.lower() in ["exit", "quit"]:
        print("\nGoodbye!")
        break

    relevant_docs = retriever.invoke(query)

    print(f"\nUser Query: {query}")

    if not relevant_docs:
        print("\nNo relevant documents found.")
        continue

    print("\n--- Retrieved Context ---")

    for i, doc in enumerate(relevant_docs, 1):
        print(f"\nDocument {i}")
        print(f"Source: {doc.metadata['source']}")
        print(doc.page_content)
        print("-" * 50)