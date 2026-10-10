
from policy_loader import load_policy_documents
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

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

MODEL_NAME = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True
)
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=False,
    return_full_text=False,
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

template = """You are a university IT support assistant.
Answer the user's question using ONLY the context below.
If the context does not contain the requested information (for example a phone number), respond that you do not know.
Never invent information, and never provide any response unrelated to the question.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def query_rag(question: str) -> str:
    return rag_chain.invoke(question)


if __name__ == "__main__":
    q = "What exact phone number should I call for the University IT Service Desk?"
    print("Q:", q)
    print("A:", query_rag(q))
