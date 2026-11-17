from flask import Flask, request, render_template, session
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers


app = Flask(__name__)
secret_key = os.getenv("FLASK_SECRET_KEY")
if not secret_key:
    raise RuntimeError("FLASK_SECRET_KEY must be set in the environment")
app.secret_key = secret_key


@app.route("/", methods=["GET", "POST"])
def index():
    if "chat" not in session:
        session["chat"] = []

    if request.method == "POST":
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        question = request.form.get("question")
        pdf_file = request.files.get("pdf")

        if token and pdf_file and question:
            pdf_path = "temp.pdf"
            pdf_file.save(pdf_path)

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
           chunk_size=300,
            chunk_overlap=50
              )
            chunks = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(chunks, embeddings)

            llm = CTransformers(
    model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    model_type="llama",
    config={
        "max_new_tokens": 80,      # 🔥 much faster
        "temperature": 0.3,
        "context_length": 512,
        "threads": 8               # 🔥 use CPU cores
    }
)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                chain_type="stuff"
            )

            answer = qa_chain.run(question)

            session["chat"].append({"q": question, "a": answer})

    return render_template("index.html", chat=session.get("chat", []))


if __name__ == "__main__":
    app.run(debug=False)
