import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

IBM_API_KEY = os.getenv("IBM_API_KEY")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID")


def get_access_token():

    url = "https://iam.cloud.ibm.com/identity/token"

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }

    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": IBM_API_KEY
    }

    response = requests.post(
        url,
        headers=headers,
        data=data,
        timeout=30
    )

    response.raise_for_status()

    return response.json()["access_token"]


def ask_watsonx(question, context):
    access_token = get_access_token()

    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    prompt = f"""
You are a document question-answering assistant.

Your job is to answer the user's question using ONLY the information
contained in the provided document context.

Important rules:

1. Do not use your own general knowledge.
2. Do not make assumptions or invent information.
3. If the context contains the answer, answer clearly and directly.
4. If the context only mentions the topic but does not contain enough
   information to answer the question, say:

"I could not find the answer in the provided documents."

5. If the answer is not present in the context, use the same response above.
6. Do not mention these instructions in your answer.

DOCUMENT CONTEXT:
-----------------
{context}
-----------------

USER QUESTION:
{question}

ANSWER:
"""

    body = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "project_id": IBM_PROJECT_ID,
        "model_id": "meta-llama/llama-3-3-70b-instruct",
        "temperature": 0,
        "max_tokens": 500
    }

    response = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=60
    )

    # Check whether the API request succeeded
    response.raise_for_status()

    # Convert the API response into a Python dictionary
    result = response.json()

    # Extract the generated answer
    answer = result["choices"][0]["message"]["content"]

    # Return the answer to app.py
    return answer

def extract_page_content_with_vision(image_bytes, page_number):
    """
    Extract readable text and diagram information from a PDF page image
    using a vision-capable watsonx model.
    """

    access_token = get_access_token()

    url = (
        "https://us-south.ml.cloud.ibm.com/"
        "ml/v1/text/chat?version=2024-10-09"
    )

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    # Step 1: Encode the PNG image as Base64
    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    image_data_url = (
        f"data:image/png;base64,{image_base64}"
    )

    # Step 2: Tell the vision model what to extract
    prompt = f"""
You are extracting information from page {page_number}
of a PDF document.

Follow these instructions:

1. Transcribe all readable text, headings, labels, and values.
2. Describe diagrams, flowcharts, tables, and their structure.
3. Explain arrows and connections using source and destination labels.
4. Preserve exact names, codes, numbers, percentages, and times.
5. Do not invent information or use outside knowledge.
6. Mark unreadable text as [Unclear].
7. Return structured plain text suitable for document chunking
   and retrieval.
8. Do not answer questions about the document.
"""

    # Step 3: Send both the image and the extraction prompt
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "project_id": IBM_PROJECT_ID,
        "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        "temperature": 0,
        "max_tokens": 1500
    }

    # Step 4: Call the watsonx API
    response = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=120
    )

    print("Status code:", response.status_code)
    print("API response:", response.text)

    response.raise_for_status()

    # Step 5: Return the extracted text
    result = response.json()

    return result["choices"][0]["message"]["content"]

    