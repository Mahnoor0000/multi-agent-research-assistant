# ============================================
#        RESEARCH ASSISTANT (Multi-Agent)
#        Search Agent + Reporter Agent + Q&A Agent
#        User Proxy Agent coordinates everything
# ============================================
from PyPDF2 import PdfReader
import requests
from ollama import chat
from autogen import AssistantAgent, UserProxyAgent

code_execution_config = {
    "use_docker": False
}



# ==========================
#   OLLAMA WRAPPER
# ==========================
def ollama_chat(prompt: str, model="phi3:mini"):
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ==========================
#   USER PROXY AGENT
# ==========================
controller_agent = UserProxyAgent(
    name="controller_agent",
    system_message=(
        "You are the Controller Agent. "
        "You coordinate all other agents (search_agent, reporter_agent, qa_agent). "
        "You route tasks correctly and manage the workflow."
    ),
code_execution_config=code_execution_config
)


# ==========================
#   SEARCH AGENT
# ==========================
search_agent = AssistantAgent(
    name="search_agent",
    system_message=(
        "You are the Search Agent.\n"
        "Your role is to retrieve relevant papers from Semantic Scholar.\n"
        "Return metadata ONLY:\n"
        "- Title\n"
        "- Authors\n"
        "- Abstract\n"
        "- URL\n"
        "Do NOT summarize or analyze."
    ),
    code_execution_config=code_execution_config

)


# ==========================
#   REPORTER AGENT (SUMMARY + REPORT)
# ==========================
reporter_agent = AssistantAgent(
    name="reporter_agent",
    system_message=(
        "You are the Reporter Agent.\n"
        "You take (Title, Authors, Abstract) and generate a short, clean, "
        "professional research report.\n\n"
        "FORMAT:\n"
        "# Research Report\n"
        "## Title\n"
        "## Authors\n"
        "## Summary\n"
        "## Key Points\n"
        "## Method\n"
        "## Limitations\n"
        "## Future Work\n\n"
        "RULES:\n"
        "- Keep it SHORT\n"
        "- No long paragraphs\n"
        "- No repetition\n"
        "- Clean, neat Markdown"
    ),
code_execution_config=code_execution_config

)
# ==========================
#   PDF SUPPORT AGENT
# ==========================

pdf_agent = AssistantAgent(
    name="pdf_agent",
    system_message=(
        "You are the PDF Extraction Agent.\n"
        "Your job is to extract readable text from PDF files.\n"
        "You DO NOT summarize. You only return cleaned, extracted text.\n"
        "If text contains irregular spacing or line breaks, fix formatting.\n"
    ),
    code_execution_config={"use_docker": False}
)

# ==========================
#   Q&A AGENT
# ==========================
qa_agent = AssistantAgent(
    name="qa_agent",
    system_message=(
        "You are the Q&A Agent.\n"
        "You answer questions using ONLY the paper abstract.\n"
        "If the answer is not found in the abstract, respond:\n"
        "'The abstract does not provide enough information to answer this question.'"
    ),
code_execution_config=code_execution_config)


# ==========================
#   SEMANTIC SCHOLAR API SEARCH
# ==========================
def search_semantic_scholar(query, max_results=5):
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={query}&limit={max_results}&fields=title,abstract,authors,url"
    )
    r = requests.get(url)
    data = r.json()

    papers = []
    for p in data.get("data", []):
        papers.append({
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "authors": [a["name"] for a in p.get("authors", [])],
            "url": p.get("url", "")
        })
    return papers


# ==========================
#   REPORT GENERATION (AGENT)
# ==========================
def generate_fast_report(paper):
    user_prompt = f"""
    Generate a short research report using ONLY:

    Title: {paper['title']}
    Authors: {', '.join(paper['authors'])}
    Abstract: {paper['abstract']}
    """
    response = reporter_agent.initiate_chat(user_message=user_prompt)
    return response["content"]


# ==========================
#   QUESTION ANSWERING (AGENT)
# ==========================
def answer_question_about_paper(paper, question):
    prompt = f"""
    Title: {paper['title']}
    Authors: {', '.join(paper['authors'])}
    Abstract: {paper['abstract']}

    Question: {question}
    """
    return ollama_chat(prompt)


# ==========================
#   PDF TEXT EXTRACTION
# ==========================

def extract_pdf_text(pdf_file):
    """
    Extracts text from a PDF uploaded in Streamlit.
    """
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    # Send to PDF agent to clean + return text
    response = pdf_agent.initiate_chat(
        recipient=pdf_agent,
        user_message=f"Clean and return this PDF-extracted text:\n\n{text}"
    )

    return response["content"]


def generate_report_from_pdf(clean_pdf_text):
    prompt = f"""
    Create a short research report from the following PDF content:

    {clean_pdf_text}

    FORMAT:
    # Research Report
    ## Summary
    ## Key Points
    ## Method
    ## Limitations
    ## Future Work
    """

    return reporter_agent.initiate_chat(
        recipient=reporter_agent,
        user_message=prompt
    )["content"]
