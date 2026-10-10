### COMPLETE THE CODE  ###


## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
from policy_loader import load_policy_documents
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

documents = load_policy_documents()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_db"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
    collection_name="university_it_policies",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


@tool
def rag_search(query: str) -> str:
    """Search the university IT policy knowledge base for relevant policy text."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant policy documents found in the knowledge base."
    return "\n\n".join(doc.page_content for doc in docs)
