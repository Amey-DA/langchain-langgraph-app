
import os
import uuid
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session
)

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "development-secret-key"
)

# Store active chat messages separately for each browser session
chat_sessions = {}


def get_messages():
    """Retrieve or initialize the current chat history."""

    chat_id = session.get("chat_id")

    if not chat_id:
        chat_id = str(uuid.uuid4())
        session["chat_id"] = chat_id

    if chat_id not in chat_sessions:
        chat_sessions[chat_id] = []

    return chat_sessions[chat_id]


@app.route("/", methods=["GET", "POST"])
def home():

    messages = get_messages()

    if request.method == "POST":

        question = request.form.get("question", "").strip()

        if not question:
            return redirect(url_for("home"))

        if len(question) > 2000:
            messages.append({
                "role": "assistant",
                "content": "Your question is too long. Please limit it to 2000 characters."
            })
            return redirect(url_for("home"))

        # Store user question
        messages.append({
            "role": "user",
            "content": question,
             "timestamp": datetime.now().strftime("%I:%M %p")
        })

        try:
            # Import only when a question is submitted
            from graph_runner import run_question

            # Execute the complete LangGraph workflow
            result = run_question(question)

            # Extract the DataFrame to display: prefer the analysis output
            df = result.get("analysis_result")
            if df is None:
                df = result.get("python_result")

            columns = []
            rows = []
            row_count = 0

            if df is not None:
                row_count = len(df)
                columns = df.columns.tolist()

                # Display only first 10 rows
                preview_df = df.head(10)

                rows = [
                    [str(value) for value in row]
                    for row in preview_df.values.tolist()
                ]

            # Store assistant response
            messages.append({
                "role": "assistant",
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "content": result.get("final_response", ""),
                "sql_query": result.get("sql_query", ""),
                "columns": columns,
                "rows": rows,
                "row_count": row_count,
                "request_mode": result.get("request_mode", ""),
                "analysis_required": result.get("analysis_required", False),
                "error": result.get("error")
            })

        except Exception as error:

            messages.append({
                "role": "assistant",
                "content": (
                    "Unable to process your question. "
                    "Please check the backend configuration."
                ),
                "sql_query": "",
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": str(error)
            })

        return redirect(url_for("home"))

    return render_template(
        "index.html",
        messages=messages
    )


@app.route("/new-chat", methods=["POST"])
def new_chat():

    # Remove the current chat history
    chat_id = session.get("chat_id")

    if chat_id:
        chat_sessions.pop(chat_id, None)

    session.pop("chat_id", None)

    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
