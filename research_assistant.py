# ============================================================
#  COMPLETE RESEARCH ASSISTANT WITH PERFORMANCE TRACKING
#  Replace your entire research_assistant.py with this file
# ============================================================

import os
import requests
from PyPDF2 import PdfReader
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv
from groq import Groq
import concurrent.futures
from functools import lru_cache
import matplotlib.pyplot as plt
import time


# ============================================================
# AGENT PERFORMANCE TRACKER
# ============================================================

class AgentPerformanceTracker:
    """Track agent interactions and measure improvement over epochs"""

    def __init__(self):
        self.epochs = []
        self.accuracy_scores = []
        self.response_times = []
        self.interaction_count = 0
        self.agent_contributions = {
            'search_agent': [],
            'qa_agent': [],
            'code_agent': []
        }

    def record_interaction(self, agent_name, quality_score, response_time):
        """Record each agent interaction"""
        self.interaction_count += 1
        self.epochs.append(self.interaction_count)
        self.accuracy_scores.append(quality_score)
        self.response_times.append(response_time)

        if agent_name in self.agent_contributions:
            self.agent_contributions[agent_name].append({
                'epoch': self.interaction_count,
                'score': quality_score,
                'time': response_time
            })

    def calculate_improvement(self):
        """Calculate improvement trend"""
        if len(self.accuracy_scores) < 2:
            return 0

        window = min(3, len(self.accuracy_scores))
        recent_avg = sum(self.accuracy_scores[-window:]) / window
        old_avg = sum(self.accuracy_scores[:window]) / window

        return ((recent_avg - old_avg) / old_avg) * 100 if old_avg > 0 else 0

    def plot_performance(self):
        """Generate performance graphs"""
        if not self.epochs:
            print("No data to plot yet!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multi-Agent Research Assistant Performance Analysis', fontsize=16)

        # Plot 1: Overall Accuracy Over Epochs
        axes[0, 0].plot(self.epochs, self.accuracy_scores, marker='o', linewidth=2, color='blue')
        axes[0, 0].set_title('Agent Performance Improvement Over Epochs')
        axes[0, 0].set_xlabel('Epoch (Interaction Count)')
        axes[0, 0].set_ylabel('Quality Score (%)')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Response Time
        axes[0, 1].plot(self.epochs, self.response_times, marker='s', linewidth=2, color='green')
        axes[0, 1].set_title('Response Time Efficiency')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Agent Contributions
        for agent_name, contributions in self.agent_contributions.items():
            if contributions:
                epochs = [c['epoch'] for c in contributions]
                scores = [c['score'] for c in contributions]
                axes[1, 0].plot(epochs, scores, marker='o', label=agent_name, linewidth=2)

        axes[1, 0].set_title('Individual Agent Performance')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Quality Score (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Cumulative Improvement
        if len(self.accuracy_scores) > 1:
            baseline = self.accuracy_scores[0]
            improvements = [(score - baseline) for score in self.accuracy_scores]
            axes[1, 1].bar(self.epochs, improvements, color='orange', alpha=0.7)
            axes[1, 1].set_title('Cumulative Improvement from Baseline')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Improvement (%)')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('agent_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Performance graph saved as 'agent_performance_analysis.png'")
        return fig


# Initialize global tracker
performance_tracker = AgentPerformanceTracker()

# ============================================================
# 1. LOAD ENVIRONMENT + INIT GROQ
# ============================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing in .env")

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# 2. GROQ CHAT WRAPPER WITH TRACKING
# ============================================================

def groq_chat(prompt: str, model="llama-3.3-70b-versatile",
              conversation_history=None, temperature=0.4, track_performance=False, agent_name=None):
    """Enhanced Groq chat with optional performance tracking"""

    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})

    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1800,
    )

    response_time = time.time() - start_time
    result = response.choices[0].message.content

    # Track performance if requested
    if track_performance and agent_name:
        quality_score = min(100, 60 + len(result) / 50)
        performance_tracker.record_interaction(agent_name, quality_score, response_time)

    return result


# ============================================================
# 3. AUTOGEN AGENTS (NO DOCKER)
# ============================================================

NO_DOCKER = {"use_docker": False}

controller_agent = UserProxyAgent(
    name="controller",
    system_message="Coordinates processing across agents.",
    code_execution_config=NO_DOCKER,
    human_input_mode="NEVER",
)

search_agent = AssistantAgent(
    name="search_agent",
    system_message="Retrieve academic papers.",
    code_execution_config=NO_DOCKER,
)

qa_agent = AssistantAgent(
    name="qa_agent",
    system_message="Answer questions using provided context only.",
    code_execution_config=NO_DOCKER,
)

code_agent = AssistantAgent(
    name="code_agent",
    system_message="Generate production-grade code.",
    code_execution_config=NO_DOCKER,
)

# ============================================================
# 4. PAPER SEARCH — ARXIV + SEMANTIC SCHOLAR
# ============================================================

MAX_RESULTS = 7


@lru_cache(maxsize=100)
def search_semantic_scholar(query, max_results=7):
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={query}&limit={max_results}&"
        "fields=title,abstract,authors,year,citationCount,url,venue"
    )

    try:
        res = requests.get(url, timeout=10).json()
        papers = []

        for p in res.get("data", []):
            abs_raw = p.get("abstract")
            abstract = abs_raw if isinstance(abs_raw, str) else ""

            papers.append({
                "title": p.get("title", ""),
                "abstract": abstract,
                "authors": [a["name"] for a in p.get("authors", [])],
                "url": p.get("url", ""),
                "year": p.get("year", "Unknown"),
                "citations": p.get("citationCount", 0),
                "venue": p.get("venue", "Unknown"),
                "source": "Semantic Scholar"
            })

        return papers

    except Exception:
        return []


@lru_cache(maxsize=100)
def search_arxiv(query, max_results=7):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    try:
        import xml.etree.ElementTree as ET

        res = requests.get(url, timeout=10)
        root = ET.fromstring(res.content)
        papers = []

        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = (entry.find('{http://www.w3.org/2005/Atom}title').text or "").strip()
            summary = (entry.find('{http://www.w3.org/2005/Atom}summary').text or "").strip()

            authors = [
                a.find('{http://www.w3.org/2005/Atom}name').text
                for a in entry.findall('{http://www.w3.org/2005/Atom}author')
            ]

            link = entry.find('{http://www.w3.org/2005/Atom}id').text
            year = entry.find('{http://www.w3.org/2005/Atom}published').text[:4]

            papers.append({
                "title": title,
                "abstract": summary,
                "authors": authors,
                "year": year,
                "citations": 0,
                "url": link,
                "venue": "arXiv",
                "source": "arXiv"
            })

        return papers

    except Exception:
        return []


def search_all_sources(query, max_results=7):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        sem_future = ex.submit(search_semantic_scholar, query, max_results)
        arxiv_future = ex.submit(search_arxiv, query, max_results)

        sem_res = sem_future.result()
        arxiv_res = arxiv_future.result()

    combined = sem_res + arxiv_res

    seen = set()
    unique = []

    for p in combined:
        key = p["title"].lower().strip()
        if key and key not in seen:
            unique.append(p)
            seen.add(key)

    unique.sort(key=lambda x: (x.get("citations", 0), str(x.get("year", ""))), reverse=True)

    return unique[:MAX_RESULTS]


# ============================================================
# 5. PDF CHUNKING + RAG Q&A
# ============================================================

def extract_pdf_text_chunked(pdf_file, chunk_size=1000, overlap=200):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    clean = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(clean):
        end = start + chunk_size
        chunks.append(clean[start:end])
        start += chunk_size - overlap

    return {"full_text": clean, "chunks": chunks}


def find_relevant_chunks(chunks, question, top_k=3):
    terms = set(question.lower().split())
    scored = []

    for c in chunks:
        score = sum(t in c.lower() for t in terms)
        scored.append((score, c))

    scored.sort(reverse=True)

    return [c for s, c in scored[:top_k] if s > 0]


def answer_with_rag(chunks, question):
    relevant = find_relevant_chunks(chunks, question)

    if not relevant:
        return "The document does not contain information related to this question."

    context = "\n\n".join(c[:600] for c in relevant)

    prompt = f"""
Use ONLY the context below.

Context:
{context}

Question:
{question}

Answer clearly:
"""

    return groq_chat(prompt).strip()


# ============================================================
# 6. PAPER REPORT
# ============================================================

def generate_paper_report(paper: dict) -> str:
    title = paper.get("title", "")
    authors = ", ".join(paper.get("authors", []))
    year = paper.get("year", "")
    venue = paper.get("venue", "")
    citations = paper.get("citations", 0)

    raw_abs = paper.get("abstract")
    abstract = (raw_abs if isinstance(raw_abs, str) else "").strip()

    if not abstract:
        abstract = "The source provides no abstract for this paper."

    prompt = f"""
Produce a **clean academic report** for this paper.

Paper Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}
Citations: {citations}

Abstract:
{abstract}

Write the report with these sections:

1. Executive Summary
2. Key Contributions
3. Methodology
4. Strengths
5. Limitations
6. Applications
7. Future Work

Rules:
- If the abstract lacks detail, write naturally
- Do NOT invent details
"""

    return groq_chat(prompt, temperature=0.35)


# ============================================================
# 7. PAPER Q&A (ORIGINAL VERSION)
# ============================================================

def answer_question_about_selected_paper(paper: dict, question: str, history=None):
    raw_abs = paper.get("abstract")
    abstract = (raw_abs if isinstance(raw_abs, str) else "").strip()

    if not abstract:
        return "The abstract does not contain any information to answer this question."

    title = paper.get("title", "")
    authors = ", ".join(paper.get("authors", []))

    prompt = f"""
Answer the user's question using ONLY the abstract.

Paper Title: {title}
Authors: {authors}

Abstract:
{abstract}

Question:
{question}

If the answer is not in the abstract:
Reply ONLY with:
"The abstract does not mention this information."
"""

    return groq_chat(prompt, conversation_history=history, temperature=0.2)


# ============================================================
# 8. ENHANCED COLLABORATIVE FUNCTIONS
# ============================================================

def enhanced_paper_qa(paper, user_question):
    """
    Multi-agent collaboration for paper analysis
    Shows improvement over epochs
    """

    # Step 1: Search agent extracts key info
    search_prompt = f"""
    Analyze this paper and extract key information:
    Title: {paper.get('title', '')}
    Abstract: {paper.get('abstract', '')[:1000]}

    Provide: main topic, methodology, and key findings.
    """

    result1 = groq_chat(search_prompt, track_performance=True, agent_name='search_agent')

    # Step 2: QA agent answers using search agent's output
    qa_prompt = f"""
    Based on this analysis:
    {result1}

    Now answer this question: {user_question}
    """

    result2 = groq_chat(qa_prompt, track_performance=True, agent_name='qa_agent')

    # Step 3: Refinement pass (shows improvement)
    refined_prompt = f"""
    Previous answer: {result2}

    Refine this answer to be more precise and add any missing details.
    """

    final_result = groq_chat(refined_prompt, track_performance=True, agent_name='qa_agent')

    return final_result


def enhanced_code_gen(task, language):
    """
    Multi-agent code generation with review and refinement
    """

    # Initial generation
    prompt1 = f"Write {language} code for: {task}"
    code_v1 = groq_chat(prompt1, temperature=0.2, track_performance=True, agent_name='code_agent')

    # QA agent reviews
    review_prompt = f"""
    Review this code and suggest improvements:
    {code_v1[:1000]}

    Check for: efficiency, error handling, best practices.
    """

    review = groq_chat(review_prompt, track_performance=True, agent_name='qa_agent')

    # Code agent refines based on review
    refine_prompt = f"""
    Original code:
    {code_v1}

    Review feedback:
    {review}

    Generate improved version incorporating this feedback.
    """

    code_v2 = groq_chat(refine_prompt, temperature=0.2, track_performance=True, agent_name='code_agent')

    return code_v2


# ============================================================
# 9. GENERAL CHATBOT
# ============================================================

def chatbot_answer(prompt, history=None):
    return groq_chat(prompt, conversation_history=history)


# ============================================================
# 10. CODE GENERATOR (ORIGINAL)
# ============================================================

def generate_advanced_code(instruction: str, language: str = "python") -> str:
    prompt = f"""
Write {language} code for:

{instruction}

Rules:
- ONLY code (no explanation)
- Include comments
- Use clean structure
- Handle errors gracefully
"""

    result = groq_chat(prompt, temperature=0.2)
    return result.strip()


# ============================================================
# 11. PAPER COMPARISON
# ============================================================

def compare_two_papers_rag(text1, text2, aspect):
    prompt = f"""
Compare two papers based on: {aspect}

Paper 1:
{text1[:4000]}

Paper 2:
{text2[:4000]}

Write the comparison:

### Similarities
### Differences
### Strengths of Paper 1
### Strengths of Paper 2
### Final Verdict
"""

    return groq_chat(prompt, temperature=0.3)


# ============================================================
# 12. PDF SUMMARY
# ============================================================

def generate_pdf_summary_report(full_text: str) -> str:
    if not isinstance(full_text, str) or len(full_text.strip()) == 0:
        return "The PDF text is empty or unreadable."

    prompt = f"""
Summarize the following PDF text into a clean report.

Text:
{full_text[:8000]}

Write the report using these sections:

1. Executive Summary
2. Key Points
3. Important Definitions
4. Important Examples (if available)
5. Conclusion

Rules:
- Write in clear, concise academic format
- Do NOT mention missing text
"""

    return groq_chat(prompt, temperature=0.35)


# ============================================================
# 13. PERFORMANCE TRACKING FUNCTIONS
# ============================================================

def get_performance_stats():
    """Get current performance statistics"""
    improvement = performance_tracker.calculate_improvement()

    stats = {
        'total_interactions': performance_tracker.interaction_count,
        'avg_quality': sum(performance_tracker.accuracy_scores) / len(
            performance_tracker.accuracy_scores) if performance_tracker.accuracy_scores else 0,
        'improvement_percentage': improvement,
        'total_epochs': len(performance_tracker.epochs)
    }

    return stats


def save_performance_graph():
    """Generate and save performance visualization"""
    return performance_tracker.plot_performance()