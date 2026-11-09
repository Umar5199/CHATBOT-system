import os
import shutil
import uuid
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global embeddings model (loaded once) ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# --- Gemini LLM (loaded once) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
    max_output_tokens=512,
    timeout=60
)

# --- Helper: index a PDF for a specific client ---
def index_pdf_for_client(pdf_path: str, client_id: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    client_folder = os.path.join("client_data", client_id)
    os.makedirs(client_folder, exist_ok=True)
    vectorstore.save_local(client_folder)
    return len(chunks)

# --- Endpoint: Upload a PDF (auto-generates client_id) ---
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    client_id: Optional[str] = Form(None)
):
    if not client_id:
        client_id = f"client_{uuid.uuid4().hex[:8]}"

    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed."}

    client_folder = os.path.join("client_data", client_id)
    os.makedirs(client_folder, exist_ok=True)
    file_path = os.path.join(client_folder, "document.pdf")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        num_chunks = index_pdf_for_client(file_path, client_id)
        return {
            "status": "success",
            "client_id": client_id,
            "chunks_indexed": num_chunks,
            "message": f"Client ID '{client_id}' created."
        }
    except Exception as e:
        return {"error": str(e)}

# --- Endpoint: Chat with a client's document ---
class Question(BaseModel):
    query: str
    client_id: str

@app.post("/chat")
def chat(q: Question):
    client_folder = os.path.join("client_data", q.client_id)
    index_file = os.path.join(client_folder, "index.faiss")

    if not os.path.exists(index_file):
        return {"error": f"No document found for client '{q.client_id}'. Please upload a PDF first."}

    vectorstore = FAISS.load_local(
        client_folder,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    answer = qa_chain.invoke({"query": q.query})
    return {"answer": answer["result"]}