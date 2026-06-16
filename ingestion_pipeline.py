import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# ----------------------------------Reading the data and verifying after reading -------------------------------------
def load_documents(data_path: str = "docs"):
    print(f"loading documents from{data_path}...")
    """
    Load all .txt files from the data directory.

    Args:
        data_path (str): Path to document folder

    Returns:
        list[Document]: Loaded LangChain documents
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f" The Directory {data_path} does not found...")

    loader = DirectoryLoader(
        path=data_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} document(s)")

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt file present...")

    # check the output
    # for i, doc in enumerate(documents[:2]):  # Show first 2 documents
    #     print(f"\nDocument {i + 1}:")
    #     print(f" Source: {doc.metadata['source']}")
    #     print(f" Content length: {len(doc.page_content)} characters")
    #     print(f" Content preview: {doc.page_content[:100]}...")
    #     print(f" Metadata: {doc.metadata}")

    return documents


# -------------------converting to chunks of data after reading --------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # print(f"Created {len(chunks)} chunks")
    # for i, chunk in enumerate(chunks[:3]):
    #     print(f"\nChunk {i+1}")
    #     print(f"Source: {chunk.metadata['source']}")
    #     print(f"Length: {len(chunk.page_content)}")
    #     print(chunk.page_content[:150])
    return chunks

# Embedding of chunks


# updating embedding as this is chargeble
# def create_embeddings():
#     print("Creating embedding model...")
#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-small"
#     )
#     return embeddings


def create_embeddings():
    print("Creating HuggingFace embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings



# Storing in DB
print("API Key Found:", bool(os.getenv("OPENAI_API_KEY")))

def create_vector_store(chunks, embeddings):
    print("Creating vector store...")

    # Generate embedding for first chunk
    # sample_vector = embeddings.embed_query(
    #     chunks[0].page_content
    # )
    # print("\nFirst Chunk:")
    # print(chunks[0].page_content[:100])
    # print("\nEmbedding Dimension:")
    # print(len(sample_vector))
    # print("\nFirst 10 Vector Values:")
    # print(sample_vector[:10])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")

    return vector_store


# Main Method calling

def main():
    documents = load_documents(data_path="docs")
    chunks = split_documents(documents)
    embeddings = create_embeddings()

    print(f"embedding data {embeddings}")

    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    main()









