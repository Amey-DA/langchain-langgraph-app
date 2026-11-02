from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import threading

from rag_pipeline import process_document
from retriever import search_documents
from llm import ask_watsonx

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "documents")

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}

document_status = {}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_uploaded_files():
    files = []

    if not os.path.exists(UPLOAD_FOLDER):
        return files

    for filename in os.listdir(UPLOAD_FOLDER):

        file_path = os.path.join(UPLOAD_FOLDER, filename)

        if os.path.isfile(file_path) and allowed_file(filename):

            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)

            status = document_status.get(
                filename,
                "Indexed"
            )

            files.append({
                "name": filename,
                "size": f"{size_mb:.2f} MB",
                "status": status
            })

    return files


@app.route("/")
def home():

    files = get_uploaded_files()

    return render_template(
        "index.html",
        files=files
    )


@app.route("/upload", methods=["POST"])
def upload():

    document = request.files.get("document")

    if not document:
        return "No file selected."

    if document.filename == "":
        return "No file selected."

    if not allowed_file(document.filename):
        return "Only PDF files are allowed."

    os.makedirs(
        UPLOAD_FOLDER,
        exist_ok=True
    )

    filename = secure_filename(
        document.filename
    )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    document.save(file_path)

    # Immediately mark as Processing
    document_status[filename] = "Processing"

    # Process document in background
    thread = threading.Thread(
        target=process_uploaded_document,
        args=(file_path, filename)
    )

    thread.start()

    return redirect(
        url_for("home")
    )


def process_uploaded_document(file_path, filename):
    try:

        document_status[filename] = "Processing"

        print("\n====================================")
        print("Processing:", filename)
        print("====================================")

        total_chunks = process_document(
            file_path,
            filename
        )

        document_status[filename] = "Indexed"

        print(
            f"Successfully indexed {total_chunks} pages."
        )

    except Exception as e:

        document_status[filename] = "Failed"

        print("\nERROR while processing document:")
        print(e)


@app.route("/document-status")
def document_status_api():

    return jsonify(document_status)


@app.route("/ask", methods=["POST"])
def ask():

    question = request.form.get(
        "question",
        ""
    ).strip()

    if not question:
        return jsonify({
            "answer": "Please enter a question.",
            "sources": []
        })

    # Handle simple greetings without running RAG
    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    }

    if question.lower() in greetings:
        return jsonify({
            "answer": "Hello! You can ask me questions about the documents you have uploaded.",
            "sources": []
        })

    try:

        # Retrieve relevant documents
        results = search_documents(
            question,
            top_k=3
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context = "\n\n".join(
            documents
        )

        answer = ask_watsonx(
            question,
            context
        )

        sources = []

        for metadata in metadatas:

            sources.append({
                "source": metadata.get(
                    "source",
                    "Unknown"
                ),
                "page": metadata.get(
                    "page_number",
                    "Unknown"
                )
            })

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:

        print("\nERROR while processing question:")
        print(e)

        return jsonify({
            "answer": "An error occurred while processing your question.",
            "sources": []
        }), 500
    
if __name__ == "__main__":
    app.run(debug=True)