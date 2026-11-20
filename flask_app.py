import os
import uuid
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

app = Flask(__name__)

# Set FLASK_SECRET_KEY in .env for anything other than local testing.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "development-secret-key")

# Chat history per browser session (kept in memory, lost on restart)
chat_sessions = {}

# CSV exports of results (one file per downloadable table)
EXPORT_DIR = Path(__file__).parent / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

PREVIEW_ROWS = 100     # rows shown in the table
MAX_MESSAGES = 50      # keep the chat history from growing forever


def cleanup_exports_on_startup():
    """
    Chat history lives in memory, so after a restart no chat owns the old CSV
    files and they can never be downloaded. Remove them.
    """
    for path in EXPORT_DIR.glob("*.csv"):
        try:
            path.unlink()
        except OSError:
            pass


cleanup_exports_on_startup()

# Seeing this line in the console confirms the NEW file is the one running.
print(f"[flask_app] loaded: table preview {PREVIEW_ROWS} rows, CSV downloads enabled")


def now() -> str:
    return datetime.now().strftime("%I:%M %p")


def get_messages():
    """Retrieve or initialize the current chat history."""

    chat_id = session.get("chat_id")

    if not chat_id:
        chat_id = str(uuid.uuid4())
        session["chat_id"] = chat_id

    if chat_id not in chat_sessions:
        chat_sessions[chat_id] = []

    return chat_sessions[chat_id]


def preview_rows(df):
    """First PREVIEW_ROWS rows as lists of strings, with empty cells for nulls."""

    preview = df.head(PREVIEW_ROWS)
    preview = preview.astype(object).where(preview.notna(), "")

    return [[str(value) for value in row] for row in preview.values.tolist()]


def add_export(downloads, df, label):
    """
    Write a DataFrame to a CSV file and add its download entry to the list.
    A failed export is logged and skipped; it never breaks the answer.
    """

    try:
        file_id = uuid.uuid4().hex
        df.to_csv(EXPORT_DIR / f"{file_id}.csv", index=False)
        downloads.append({"id": file_id, "label": label})
    except Exception as error:
        print(f"[EXPORT FAILED] {label}: {error}")


def delete_exports(messages):
    """Remove the CSV files that belong to the given messages."""

    for message in messages:
        for entry in message.get("downloads", []):
            path = EXPORT_DIR / f"{entry['id']}.csv"
            if path.exists():
                path.unlink()


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
                "timestamp": now(),
                "content": "Your question is too long. Please limit it to 2000 characters."
            })
            return redirect(url_for("home"))

        # Store user question
        messages.append({
            "role": "user",
            "content": question,
            "timestamp": now()
        })

        try:
            # Import only when a question is submitted
            from graph_runner import run_question

            # Execute the complete LangGraph workflow
            result = run_question(question)

            analysis_df = result.get("analysis_result")
            processed_df = result.get("python_result")
            sql_df = result.get("sql_result")

            # The table shows the analysis output when it exists
            df = analysis_df if analysis_df is not None else processed_df

            columns = []
            rows = []
            row_count = 0
            downloads = []

            if df is not None:
                row_count = len(df)
                columns = df.columns.tolist()
                rows = preview_rows(df)

                # The larger dataset that the displayed table was reduced from
                full_df = None
                for candidate in (processed_df, sql_df):
                    if candidate is not None and len(candidate) > row_count:
                        full_df = candidate
                        break

                if row_count > 0:
                    if row_count > PREVIEW_ROWS:
                        add_export(downloads, df, f"Download all {row_count} rows (CSV)")
                    elif full_df is not None:
                        add_export(downloads, df, "Download result (CSV)")

                if full_df is not None:
                    add_export(
                        downloads,
                        full_df,
                        f"Download full data, {len(full_df)} rows (CSV)",
                    )

            print(f"[DOWNLOADS] {len(downloads)} export(s) created for this answer")

            # Store assistant response
            messages.append({
                "role": "assistant",
                "timestamp": now(),
                "content": result.get("final_response", ""),
                "sql_query": result.get("sql_query", ""),
                "columns": columns,
                "rows": rows,
                "row_count": row_count,
                "downloads": downloads,
                "request_mode": result.get("request_mode", ""),
                "analysis_required": result.get("analysis_required", False),
                "error": result.get("error")
            })

        except Exception as error:

            messages.append({
                "role": "assistant",
                "timestamp": now(),
                "content": (
                    "Unable to process your question. "
                    "Please check the backend configuration."
                ),
                "sql_query": "",
                "columns": [],
                "rows": [],
                "row_count": 0,
                "downloads": [],
                "request_mode": "",
                "analysis_required": False,
                "error": str(error)
            })

        # Keep only the most recent messages, and delete their export files
        if len(messages) > MAX_MESSAGES:
            delete_exports(messages[:-MAX_MESSAGES])
            del messages[:-MAX_MESSAGES]

        return redirect(url_for("home"))

    return render_template(
        "index.html",
        messages=messages
    )


@app.route("/download/<file_id>")
def download(file_id):
    """Send a CSV export, only to the chat that created it."""

    owned = any(
        entry["id"] == file_id
        for message in get_messages()
        for entry in message.get("downloads", [])
    )

    path = EXPORT_DIR / f"{file_id}.csv"

    if not owned or not path.exists():
        abort(404)

    return send_file(
        path,
        as_attachment=True,
        download_name="results.csv",
        mimetype="text/csv",
    )


@app.route("/new-chat", methods=["POST"])
def new_chat():

    # Remove the current chat history and its export files
    chat_id = session.get("chat_id")

    if chat_id:
        delete_exports(chat_sessions.get(chat_id, []))
        chat_sessions.pop(chat_id, None)

    session.pop("chat_id", None)

    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)