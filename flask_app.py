import os
import uuid
from datetime import datetime
from io import BytesIO

import pandas as pd
from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)


# ------------------------------------------------------------
# Flask configuration
# ------------------------------------------------------------

app = Flask(__name__)

# Flask uses this key to sign the browser's session cookie.
# For local use, a temporary key is generated if none is configured.
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)


# ------------------------------------------------------------
# Chat and table settings
# ------------------------------------------------------------

PAGE_SIZE = 20          # Number of records visible on one page.
BROWSE_LIMIT = 20_000   # Maximum records a user can browse per result table.
MAX_MESSAGES = 50       # Maximum individual messages kept in each chat.

# The browser cookie holds only the chat ID.
# The actual messages and result DataFrames stay in this Flask process.
chat_sessions = {}


# ------------------------------------------------------------
# Chat helpers
# ------------------------------------------------------------

def now() -> str:
    """Return the time shown beside a chat message."""
    return datetime.now().strftime("%I:%M %p")


def get_messages():
    """Get the current browser's chat history, creating it when needed."""
    chat_id = session.get("chat_id")

    if not chat_id:
        chat_id = uuid.uuid4().hex
        session["chat_id"] = chat_id

    return chat_sessions.setdefault(chat_id, [])


def add_message(messages, message):
    """Add a message and discard old messages beyond the history limit."""
    messages.append(message)

    # Each assistant message owns its tables. Removing the message also
    # removes those tables from this chat's available results.
    del messages[:-MAX_MESSAGES]


# ------------------------------------------------------------
# Result table helpers
# ------------------------------------------------------------

def make_table(title, df):
    """Attach a completed DataFrame to an answer and prepare its first page."""
    row_count = len(df)
    browse_count = min(row_count, BROWSE_LIMIT)

    table = {
        "id": uuid.uuid4().hex,
        "title": title,
        "df": df,
        "columns": [str(column) for column in df.columns],
        "row_count": row_count,
        "browse_count": browse_count,
        "page_count": (browse_count + PAGE_SIZE - 1) // PAGE_SIZE,
        "can_export": row_count > BROWSE_LIMIT,
    }

    # The first page is included in the initial HTML. Later pages are
    # requested from Flask only when the user clicks Previous or Next.
    table["first_page"] = page_data(table, 1)
    return table


def page_data(table, page):
    """Read one page from the saved result and prepare it for the browser."""
    start = (page - 1) * PAGE_SIZE
    stop = min(start + PAGE_SIZE, table["browse_count"])

    # iloc slices by row position, even when Pandas kept an old index
    # after sorting or filtering the result.
    frame = table["df"].iloc[start:stop]
    frame = frame.astype(object).where(frame.notna(), "")

    return {
        "page": page,
        "first_row": start + 1 if len(frame) else 0,
        "last_row": stop,
        "rows": [
            [str(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }


def result_tables(result):
    """Choose the final tables that belong to the user's answer."""
    if result.get("error"):
        return []

    analysis_df = result.get("analysis_result")
    processed_df = result.get("python_result")
    sql_df = result.get("sql_result")

    # When analysis produced a table, that is the most recent result.
    # Otherwise use the Pandas result, or SQL for a SQL-only answer.
    # An empty DataFrame is still a valid result with zero matching rows.
    final_df = next(
        (
            frame for frame in (analysis_df, processed_df, sql_df)
            if isinstance(frame, pd.DataFrame)
        ),
        None,
    )

    if final_df is None:
        return []

    # Optional two-table path from the earlier planner change.
    # Its contract is that SQL contains the COMPLETE calculated breakdown
    # and the final DataFrame contains its requested shortlist.
    # Ordinary top-N questions should leave shortlist_size unset.
    shortlist_size = result.get("shortlist_size")
    if (
        type(shortlist_size) is int
        and shortlist_size > 0
        and isinstance(sql_df, pd.DataFrame)
        and final_df is not sql_df
        and len(final_df) == min(shortlist_size, len(sql_df))
        and len(sql_df) > len(final_df)
    ):
        return [
            make_table(f"Top {shortlist_size}", final_df),
            make_table("Full breakdown", sql_df),
        ]

    return [make_table("Results", final_df)]


def get_owned_table(table_id):
    """Find a table only within the current browser session's chat."""
    chat_id = session.get("chat_id")

    for message in chat_sessions.get(chat_id, []):
        for table in message.get("tables", []):
            if table["id"] == table_id:
                return table

    # This also covers expired tables and tables removed by New Chat.
    abort(404)


# ------------------------------------------------------------
# Home route — display the chat or process a new question
# ------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    messages = get_messages()

    if request.method == "GET":
        return render_template("index.html", messages=messages)

    question = request.form.get("question", "").strip()

    if not question:
        return redirect(url_for("home"))

    if len(question) > 2000:
        add_message(messages, {
            "role": "assistant",
            "timestamp": now(),
            "content": "Your question is too long. Please limit it to 2000 characters.",
            "tables": [],
        })
        return redirect(url_for("home"))

    add_message(messages, {
        "role": "user",
        "timestamp": now(),
        "content": question,
    })

    try:
        # Import on submission so opening the chat page does not initialize
        # the model. This is the only route that runs the analytics workflow.
        from graph_runner import run_question

        result = run_question(question)

        add_message(messages, {
            "role": "assistant",
            "timestamp": now(),
            "content": result.get("final_response", ""),
            "sql_query": result.get("sql_query", ""),
            "tables": result_tables(result),
            "request_mode": result.get("request_mode", ""),
            "analysis_required": result.get("analysis_required", False),
            "error": result.get("error"),
        })

    except Exception:
        # Record the detailed exception in the Flask console.
        app.logger.exception("Unable to process the analytics question")

        add_message(messages, {
            "role": "assistant",
            "timestamp": now(),
            "content": (
                "Unable to process your question. "
                "Please check the Flask console for details."
            ),
            "tables": [],
        })

    # Redirect after POST so refreshing the page does not rerun the question.
    return redirect(url_for("home"))


# ------------------------------------------------------------
# Pagination route — return one page of an existing result
# ------------------------------------------------------------

@app.get("/results/<table_id>/page")
def result_page(table_id):
    table = get_owned_table(table_id)
    page = request.args.get("page", default=1, type=int)

    # An empty table has an empty first page. Requests beyond the
    # browsable range must never expose rows past the 20,000-row limit.
    if page < 1 or page > max(1, table["page_count"]):
        abort(404)

    response = jsonify(page_data(table, page))
    response.cache_control.no_store = True
    return response


# ------------------------------------------------------------
# CSV route — export the complete final table when it exceeds the cap
# ------------------------------------------------------------

@app.get("/results/<table_id>/csv")
def result_csv(table_id):
    table = get_owned_table(table_id)

    if not table["can_export"]:
        abort(404)

    # Create the CSV only when someone clicks Export.
    # utf-8-sig helps Excel open names containing accents correctly.
    buffer = BytesIO()
    table["df"].to_csv(buffer, index=False, encoding="utf-8-sig")
    buffer.seek(0)

    response = send_file(
        buffer,
        as_attachment=True,
        download_name=f"results-{table_id[:8]}.csv",
        mimetype="text/csv",
    )
    response.cache_control.no_store = True
    return response


# ------------------------------------------------------------
# New Chat route — clear this chat and its saved result tables
# ------------------------------------------------------------

@app.post("/new-chat")
def new_chat():
    chat_id = session.pop("chat_id", None)

    if chat_id:
        chat_sessions.pop(chat_id, None)

    return redirect(url_for("home"))


# ------------------------------------------------------------
# Local entry point — start Flask on this computer
# ------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, use_reloader=False)