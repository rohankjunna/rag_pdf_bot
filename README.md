# PDF Question Answering AI (RAG) – Flask


## Features
- Upload any PDF
- Ask questions based only on document content
- Retrieval‑Augmented Generation (RAG)
- Clean Flask UI with styling


## Tech Stack
- Python, Flask
- LangChain
- FAISS Vector Database
- Hugging Face LLM


## Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and set both values. Generate a strong Flask
   secret key, for example with `python -c "import secrets; print(secrets.token_hex(32))"`.
3. Run the app:
   ```bash
   python app.py
   ```

Open: http://127.0.0.1:5000
"""