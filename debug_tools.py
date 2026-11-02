# debug_layers.py
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

KEY, PID = os.getenv("WATSONX_PROJECT_KEY"), os.getenv("WATSONX_PROJECT_ID")
URL = "https://us-south.ml.cloud.ibm.com"
MODELS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    "meta-llama/llama-3-3-70b-instruct",   # comparison; use any model on your list
]
ASK = "Count the rows in dbo.DM_PRODUCT using the tool."

tool_spec = {"type": "function", "function": {
    "name": "sql_query", "description": "Run a read-only SQL SELECT.",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}

# Layer 1: IBM SDK directly (no LangChain)
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
client = APIClient(Credentials(url=URL, api_key=KEY))

for mid in MODELS:
    try:
        m = ModelInference(model_id=mid, api_client=client, project_id=PID)
        msg = m.chat(messages=[{"role": "user", "content": ASK}],
                     tools=[tool_spec])["choices"][0]
        print(f"[SDK ] {mid}\n   finish={msg.get('finish_reason')} "
              f"tool_calls={msg['message'].get('tool_calls')}\n   "
              f"content={str(msg['message'].get('content'))[:80]!r}")
    except Exception as e:
        print(f"[SDK ] {mid} ERROR: {e}")

# Layer 2: langchain_ibm with the same simple tool
from langchain_ibm import ChatWatsonx
from langchain.tools import tool

@tool
def sql_query(query: str) -> str:
    """Run a read-only SQL SELECT."""
    return "ok"

for mid in MODELS:
    try:
        llm = ChatWatsonx(model_id=mid, url=URL, api_key=KEY, project_id=PID,
                          params={"temperature": 0, "max_tokens": 500})
        r = llm.bind_tools([sql_query]).invoke(ASK)
        print(f"[LC  ] {mid}\n   tool_calls={r.tool_calls} "
              f"content={str(r.content)[:80]!r}")
    except Exception as e:
        print(f"[LC  ] {mid} ERROR: {e}")