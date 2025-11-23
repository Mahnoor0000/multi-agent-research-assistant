# ============================================
#        RESEARCH ASSISTANT (Multi-Agent)
#   Search + Reporter + PDF + Q&A + Code Gen
#   Backend logic using Groq (llama-3.3-70b-versatile)
# ============================================

import os
import requests
from PyPDF2 import PdfReader
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv
from groq import Groq

# ============================================
#        LOAD ENV + INIT GROQ CLIENT
# ============================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "❌ GROQ_API_KEY not found. Please set it as an environment variable."
    )

client = Groq(api_key=GROQ_API_KEY)


# ============================================
#        GROQ CHAT WRAPPER
# ============================================
def groq_chat(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Simple wrapper around Groq chat completions.
    All agents call this function under the hood.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=900,
    )
    return response.choices[0].message.content


# ============================================
#        LIGHT AUTOGEN AGENT OBJECTS
# (for conceptual multi-agent architecture)
# ============================================
code_execution_config = {"use_docker": False}

controller_agent = UserProxyAgent(
    name="controller_agent",
    system_message="You coordinate all other agents.",
    code_execution_config=code_execution_config,
)

search_agent = AssistantAgent(
    name="search_agent",
    system_message="You retrieve metadata of research papers.",
    code_execution_config=code_execution_config,
)

reporter_agent = AssistantAgent(
    name="reporter_agent",
    system_message="You generate structured research reports.",
    code_execution_config=code_execution_config,
)

pdf_agent = AssistantAgent(
    name="pdf_agent",
    system_message="You handle PDF text extraction and cleaning.",
    code_execution_config=code_execution_config,
)

qa_agent = AssistantAgent(
    name="qa_agent",
    system_message="You answer questions about papers.",
    code_execution_config=code_execution_config,
)

code_agent = AssistantAgent(
    name="code_agent",
    system_message="You generate clean, runnable code.",
    code_execution_config=code_execution_config,
)


# ============================================
#         SEMANTIC SCHOLAR SEARCH
# ============================================
def search_semantic_scholar(query: str, max_results: int = 5):
    """
    Search Semantic Scholar for papers.
    Returns a list of dicts: {title, abstract, authors, url}.
    """
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={query}&limit={max_results}&fields=title,abstract,authors,url"
    )

    r = requests.get(url)
    data = r.json()

    papers = []
    for p in data.get("data", []):
        papers.append(
            {
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "authors": [a["name"] for a in p.get("authors", [])],
                "url": p.get("url", ""),
            }
        )
    return papers


# ============================================
#         REPORTER AGENT — FROM METADATA
# ============================================
def generate_fast_report(paper: dict) -> str:
    """
    Generate a short, structured research report
    using only title, authors, and abstract.
    """
    prompt = f"""
You are the Reporter Agent in a multi-agent research assistant.

Generate a SHORT, structured research report from this metadata:

Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Abstract: {paper['abstract']}

FORMAT:
# Research Report
## Summary (3–5 lines)
## Key Points (bullet list)
## Method (1–2 lines)
## Limitations (bullet list)
## Future Work (bullet list)

Keep it concise and academic.
"""
    return groq_chat(prompt)


# ============================================
#        Q&A AGENT — FROM PAPER ABSTRACT
# ============================================
def answer_question_about_paper(paper: dict, question: str) -> str:
    """
    Answer a question using ONLY the paper abstract.
    """
    prompt = f"""
You are the Q&A Agent in a multi-agent research assistant.
Use ONLY the abstract to answer the question.

Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Abstract: {paper['abstract']}

Question: {question}

If the answer is not clearly present in the abstract, reply:
"The abstract does not provide enough information to answer this question."
"""
    return groq_chat(prompt)


# ============================================
#             PDF TEXT EXTRACTION
# ============================================
def extract_pdf_text(pdf_file) -> str:
    """
    Extract raw text from an uploaded PDF file.
    Basic cleaning is done (whitespace collapse).
    """
    reader = PdfReader(pdf_file)
    raw_text = ""

    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            raw_text += txt + "\n"

    # Simple cleaning: collapse multiple spaces/newlines
    cleaned = " ".join(raw_text.split())
    return cleaned


# ============================================
#     REPORTER AGENT — FROM PDF CONTENT
# ============================================
def generate_report_from_pdf(clean_pdf_text: str) -> str:
    """
    Generate a report from full PDF text.
    """
    prompt = f"""
You are the Reporter Agent in a multi-agent research assistant.

Create a SHORT research report from the following PDF content:

\"\"\"{clean_pdf_text[:15000]}\"\"\"  # (context truncated if huge)

FORMAT:
# Research Report
## Summary (3–5 lines)
## Key Points (bullet list)
## Method
## Limitations
## Future Work

Be concise and avoid repetition.
"""
    return groq_chat(prompt)


# ============================================
#     Q&A AGENT — FROM PDF CONTENT (OPTIONAL)
# ============================================
def answer_question_about_pdf(clean_pdf_text: str, question: str) -> str:
    """
    Answer a question using ONLY the PDF text.
    """
    context = clean_pdf_text[:15000]  # safety truncation

    prompt = f"""
You are the Q&A Agent in a multi-agent research assistant.
Answer the user's question using ONLY the following PDF text:

\"\"\"{context}\"\"\"

Question: {question}

If the answer is not present in the text, reply:
"The PDF text does not provide enough information to answer this question."
"""
    return groq_chat(prompt)


# ============================================
#         CODE GENERATION AGENT
# ============================================
def generate_code(code_instruction: str) -> str:
    """
    Generate code using Groq (Code Agent behavior).
    Returns code only, wrapped in ``` blocks when possible.
    """
    prompt = f"""
You are the Code Generation Agent in a multi-agent research assistant.

User instruction:
\"\"\"{code_instruction}\"\"\"

Write clean, correct, runnable code.
- Use appropriate language syntax.
- If the user does not specify language, default to Python.
- Return ONLY code, wrapped in ```language``` fences.
Do not add explanations unless explicitly asked.
"""
    return groq_chat(prompt)
