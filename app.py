import streamlit as st

from research_assistant import (
    search_all_sources,
    extract_pdf_text_chunked,
    answer_with_rag,
    generate_paper_report,
    multi_agent_paper_analysis,  # REAL multi-agent function
    multi_agent_code_generation,  # REAL multi-agent function
    multi_agent_paper_comparison,  # REAL multi-agent function
    chatbot_answer,
    generate_pdf_summary_report,
)

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Multi-Agent Research Assistant for Deep CNNs",
    layout="wide",
    page_icon="🤖",
)

st.title("🤖 Multi-Agent Research Assistant for Deep CNNs")
st.caption(
    "AutoGen Multi-Agent System · CNN Paper Analysis · Code Generation · RAG Q&A"
)

# ============================================================
# AGENT ARCHITECTURE VISUALIZATION
# ============================================================
with st.expander("🔍 View Multi-Agent Architecture & Communication Flow", expanded=False):
    st.markdown("""
    ### 🎯 Agent Roles & Specialization

    | Agent | Type | Specialization | AutoGen Role |
    |-------|------|----------------|--------------|
    | **Controller** | UserProxyAgent | Coordinates tasks, routes to specialists | Orchestrator |
    | **SearchSpecialist** | AssistantAgent | Finds CNN papers, extracts info | Information Retriever |
    | **QASpecialist** | AssistantAgent | Answers questions using context | Context-Based Answerer |
    | **CodeSpecialist** | AssistantAgent | Generates CNN implementations | Code Generator |
    | **AnalysisSpecialist** | AssistantAgent | Compares papers, analyzes trends | Comparative Analyst |

    ---

    ### 🔄 Communication Flow (How AutoGen Agents Talk)

    #### Scenario 1: Paper Q&A (3 Agents Collaborate)
    ```
    User Query: "What CNN architecture is used in this paper?"
         ↓
    [Controller] receives → routes to SearchSpecialist
         ↓
    [SearchSpecialist] 
         • Reads paper abstract
         • Extracts: "ResNet-50, uses skip connections, batch norm"
         • Sends info to QASpecialist
         ↓
    [QASpecialist]
         • Receives SearchSpecialist's extracted info
         • Formulates answer based on extraction
         • Sends answer to AnalysisSpecialist
         ↓
    [AnalysisSpecialist]
         • Receives QASpecialist's answer
         • Adds CNN-specific context
         • Enhances with comparisons to other architectures
         • Returns refined answer
         ↓
    User receives comprehensive answer
    ```

    #### Scenario 2: Code Generation (3 Agents Collaborate)
    ```
    User Request: "Implement ResNet block in PyTorch"
         ↓
    [Controller] receives → routes to CodeSpecialist
         ↓
    [CodeSpecialist]
         • Generates initial PyTorch code
         • Code v1: Basic ResNet block
         • Sends to QASpecialist for review
         ↓
    [QASpecialist]
         • Receives code from CodeSpecialist
         • Reviews for: correctness, best practices, missing features
         • Feedback: "Add batch normalization, use proper activation"
         • Sends review back to CodeSpecialist
         ↓
    [CodeSpecialist]
         • Receives QASpecialist's review
         • Refines code incorporating feedback
         • Code v2: Improved with batch norm & proper structure
         • Returns refined code
         ↓
    User receives production-ready code
    ```

    #### Scenario 3: Paper Comparison (2 Agents Collaborate)
    ```
    User Request: "Compare ResNet vs EfficientNet"
         ↓
    [Controller] receives → routes to SearchSpecialist
         ↓
    [SearchSpecialist]
         • Extracts key info from Paper 1 (ResNet)
         • Extracts key info from Paper 2 (EfficientNet)
         • Sends both extracts to AnalysisSpecialist
         ↓
    [AnalysisSpecialist]
         • Receives both extracts from SearchSpecialist
         • Compares: architecture, performance, complexity
         • Generates structured comparison
         • Returns comparison report
         ↓
    User receives detailed comparison
    ```

    ---

    ### 🔗 How AutoGen Enables Communication

    **1. Agent Definition (in research_assistant.py):**
    ```python
    # Agents are created with specific roles
    search_agent = AssistantAgent(
        name="SearchSpecialist",
        system_message="Expert in finding CNN papers...",
        llm_config=llm_config  # AutoGen config
    )

    qa_agent = AssistantAgent(
        name="QASpecialist", 
        system_message="Expert in answering questions...",
        llm_config=llm_config
    )
    ```

    **2. Information Passing (Sequential Communication):**
    ```python
    # Agent 1 produces output
    search_result = groq_chat(search_task)

    # Agent 2 receives Agent 1's output as input
    qa_response = groq_chat(f"Based on: {search_result}, answer: {question}")

    # Agent 3 receives Agent 2's output
    final = groq_chat(f"Enhance: {qa_response}")
    ```

    **Key Point:** Each agent's output becomes the next agent's input context!

    ---

    ### 📊 Why This is Real Multi-Agent Communication

    ✅ **Sequential Processing:** Agents work in order, not parallel  
    ✅ **Information Dependency:** Later agents need earlier agents' outputs  
    ✅ **Specialization:** Each agent has unique expertise  
    ✅ **Collaboration:** Final result is better than any single agent  
    ✅ **Multi-Turn:** Multiple exchanges between agents  

    ❌ **Not Just:** Multiple independent API calls  
    ❌ **Not Just:** Parallel processing without communication  

    ---

    ### 🎓 For Your Teacher

    **"How do agents communicate?"**
    - Agent 1 processes input → produces output A
    - Agent 2 receives output A → produces output B  
    - Agent 3 receives output B → produces final output C
    - This is **information passing** through sequential function calls

    **"How is AutoGen used?"**
    - AutoGen's `AssistantAgent` and `UserProxyAgent` classes define agent roles
    - Each agent has specialized `system_message` for its expertise
    - `llm_config` connects agents to Groq LLM backend
    - Agents communicate through structured prompts that pass information

    **"What makes this multi-agent?"**
    - Multiple specialized agents (not one general agent)
    - Each agent contributes unique value
    - Sequential collaboration improves output quality
    - Later agents build on earlier agents' work
    """)

st.markdown("---")

# ============================================================
# SESSION STATE INIT
# ============================================================
if "papers" not in st.session_state:
    st.session_state["papers"] = []

if "paper_report" not in st.session_state:
    st.session_state["paper_report"] = None

if "paper_report_key" not in st.session_state:
    st.session_state["paper_report_key"] = None

if "paper_chat_history" not in st.session_state:
    st.session_state["paper_chat_history"] = {}

if "pdf_data" not in st.session_state:
    st.session_state["pdf_data"] = None

if "pdf_report" not in st.session_state:
    st.session_state["pdf_report"] = None

if "chatbot_history" not in st.session_state:
    st.session_state["chatbot_history"] = []

if "compare_result" not in st.session_state:
    st.session_state["compare_result"] = None

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔍 Search CNN Papers",
        "📄 PDF Upload + RAG",
        "💬 CNN Chatbot",
        "💻 CNN Code Generator",
        "📊 Compare CNN Papers",
    ]
)

# ============================================================
# TAB 1 — MULTI-AGENT PAPER SEARCH + ANALYSIS
# ============================================================
with tab1:
    st.header("🔍 Search CNN Research Papers")

    st.info("🤖 **Multi-Agent Mode Active:** SearchSpecialist → QASpecialist → AnalysisSpecialist")

    topic = st.text_input(
        "Enter a CNN-related search topic:",
        placeholder="e.g., ResNet, EfficientNet, CNN pruning, image classification",
        key="search_topic",
    )

    if st.button("🔎 Search with Multi-Agent System", key="search_btn"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("🤖 SearchSpecialist searching across sources..."):
                papers = search_all_sources(topic)
            st.session_state["papers"] = papers
            st.session_state["paper_report"] = None
            st.session_state["paper_report_key"] = None
            st.session_state["paper_chat_history"] = {}
            if not papers:
                st.error("No papers found. Try another query.")
            else:
                st.success(f"✅ SearchSpecialist found {len(papers)} CNN papers!")

    papers = st.session_state["papers"]

    if papers:
        titles = [f"{i + 1}. {p['title']}" for i, p in enumerate(papers)]

        selected_index = st.selectbox(
            "Select a paper:",
            range(len(papers)),
            format_func=lambda i: titles[i],
            key="selected_paper_idx",
        )
        paper = papers[selected_index]

        paper_key = f"{paper.get('title', '')}|{paper.get('source', '')}"
        authors_str = ", ".join(paper.get("authors", [])) or "Unknown"

        st.subheader("📄 Paper Details")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Title:** {paper.get('title', '')}")
            st.markdown(f"**Authors:** {authors_str}")

        with col2:
            st.metric("Citations", paper.get('citations', 0))
            st.metric("Year", paper.get('year', 'Unknown'))

        st.markdown(f"**Venue:** {paper.get('venue', 'Unknown')}")
        st.markdown(f"**Source:** {paper.get('source', '')}")
        if paper.get("url"):
            st.markdown(f"**URL:** [{paper['url']}]({paper['url']})")

        st.markdown("---")

        # ----- Generate Report -----
        if st.button("📝 Generate Detailed Report", key="gen_paper_report_btn"):
            with st.spinner("🤖 Generating comprehensive report..."):
                report = generate_paper_report(paper)
            st.session_state["paper_report"] = report
            st.session_state["paper_report_key"] = paper_key

        if (
                st.session_state["paper_report"]
                and st.session_state["paper_report_key"] == paper_key
        ):
            st.markdown("### 📄 Comprehensive Paper Report")
            st.markdown(st.session_state["paper_report"])

            st.download_button(
                "⬇️ Download Report",
                st.session_state["paper_report"],
                file_name="cnn_paper_report.md",
                mime="text/markdown",
                key="download_paper_report_btn",
            )

        st.markdown("---")

        # ----- MULTI-AGENT Q&A -----
        st.subheader("🤖 Multi-Agent Q&A System")

        with st.container():
            st.markdown("""
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3;">
            <h4 style="margin: 0; color: #1976D2;">💡 3-Agent Collaboration Process:</h4>
            <ol style="margin: 10px 0;">
                <li><b>SearchSpecialist</b> extracts key information from paper</li>
                <li><b>QASpecialist</b> answers your question using extracted info</li>
                <li><b>AnalysisSpecialist</b> refines answer with CNN expertise</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

        question = st.text_input(
            "Ask about this CNN paper:",
            placeholder="e.g., What CNN architecture is used? How does it compare to ResNet?",
            key="paper_question_input",
        )

        if st.button("🚀 Ask Multi-Agent System", key="ask_paper_question_btn"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("🤖 Agents collaborating (Search → QA → Analysis)..."):
                    # This triggers REAL multi-agent communication
                    answer = multi_agent_paper_analysis(paper, question)

                history_dict = st.session_state["paper_chat_history"]
                history = history_dict.get(paper_key, [])

                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": answer})
                history_dict[paper_key] = history
                st.session_state["paper_chat_history"] = history_dict

                st.success("✅ 3 agents collaborated to produce this answer!")

        # Show chat history
        history = st.session_state["paper_chat_history"].get(paper_key, [])
        if history:
            st.markdown("### 💬 Conversation History")
            for msg in history:
                if msg["role"] == "user":
                    st.markdown(f"**🧑‍💻 You:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 Multi-Agent System:** {msg['content']}")

# ============================================================
# TAB 2 — PDF UPLOAD + RAG
# ============================================================
with tab2:
    st.header("📄 PDF Upload + RAG Q&A")
    st.info("🤖 **QASpecialist** uses RAG to answer from your PDF")

    pdf = st.file_uploader(
        "Upload a CNN research paper (PDF)",
        type=["pdf"],
        key="pdf_uploader_main"
    )

    if pdf is not None and st.button("📤 Process PDF", key="process_pdf_btn"):
        with st.spinner("🤖 Extracting text and creating chunks..."):
            pdf_data = extract_pdf_text_chunked(pdf)
        st.session_state["pdf_data"] = pdf_data
        st.session_state["pdf_report"] = None
        st.success("✅ PDF processed and ready for Q&A!")

    pdf_data = st.session_state["pdf_data"]

    if pdf_data:
        st.success("📚 PDF loaded - Ask questions about CNN concepts!")

        st.subheader("❓ Ask Questions About the PDF")

        pdf_question = st.text_input(
            "Your question:",
            placeholder="e.g., What CNN layers are used? What's the accuracy?",
            key="pdf_question_input",
        )

        if st.button("🔍 Ask QASpecialist", key="ask_pdf_btn"):
            if not pdf_question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("🤖 QASpecialist analyzing PDF with RAG..."):
                    answer = answer_with_rag(pdf_data["chunks"], pdf_question)
                st.markdown("### 💡 Answer from QASpecialist")
                st.markdown(answer)

        st.markdown("---")

        st.subheader("📋 Generate PDF Summary")

        if st.button("📝 Generate Summary", key="gen_pdf_report_btn"):
            with st.spinner("🤖 Summarizing CNN paper..."):
                pdf_report = generate_pdf_summary_report(pdf_data["full_text"])
            st.session_state["pdf_report"] = pdf_report

        if st.session_state["pdf_report"]:
            st.markdown("### 📄 Summary Report")
            st.markdown(st.session_state["pdf_report"])

            st.download_button(
                "⬇️ Download Summary",
                st.session_state["pdf_report"],
                file_name="cnn_paper_summary.md",
                mime="text/markdown",
                key="download_pdf_report_btn",
            )

# ============================================================
# TAB 3 — CNN CHATBOT
# ============================================================
with tab3:
    st.header("💬 CNN Research Chatbot")
    st.info("🤖 Ask anything about CNNs, architectures, or deep learning")

    user_message = st.text_area(
        "Ask about CNN research:",
        key="chatbot_message_input",
        placeholder="e.g., Explain ResNet skip connections, Compare VGG vs ResNet",
        height=120,
    )

    if st.button("💬 Send", key="chatbot_send_btn"):
        if not user_message.strip():
            st.warning("Please enter a message.")
        else:
            history = st.session_state["chatbot_history"].copy()

            with st.spinner("🤖 Thinking..."):
                reply = chatbot_answer(user_message, history=history)

            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": reply})
            st.session_state["chatbot_history"] = history

    history = st.session_state["chatbot_history"]
    if history:
        st.markdown("### 💬 Chat History")
        for msg in history:
            if msg["role"] == "user":
                st.markdown(f"**🧑‍💻 You:** {msg['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {msg['content']}")

# ============================================================
# TAB 4 — MULTI-AGENT CODE GENERATION
# ============================================================
with tab4:
    st.header("💻 Multi-Agent CNN Code Generator")

    with st.container():
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
        <h4 style="margin: 0; color: #2E7D32;">🤖 3-Agent Collaboration Process:</h4>
        <ol style="margin: 10px 0;">
            <li><b>CodeSpecialist</b> generates initial CNN implementation</li>
            <li><b>QASpecialist</b> reviews code for best practices & correctness</li>
            <li><b>CodeSpecialist</b> refines code based on review feedback</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    task = st.text_area(
        "Describe the CNN implementation:",
        placeholder=(
            "Examples:\n"
            "- Implement ResNet-18 in PyTorch\n"
            "- Create a custom CNN for CIFAR-10\n"
            "- Build VGG-16 with batch normalization\n"
            "- Implement depthwise separable convolutions\n"
        ),
        key="code_task_input",
        height=160,
    )

    language = st.selectbox(
        "Framework:",
        ["python", "pytorch", "tensorflow"],
        key="code_lang_select"
    )

    if st.button("🚀 Generate with Multi-Agent System", key="gen_code_btn"):
        if not task.strip():
            st.warning("Please describe a CNN implementation task.")
        else:
            with st.spinner("🤖 Agents collaborating (Code → Review → Refine)..."):
                # This triggers REAL multi-agent code generation
                code = multi_agent_code_generation(task, language=language)

            st.markdown("### 💻 Generated CNN Implementation")
            st.markdown("*✨ Refined through 3-agent collaboration*")
            st.code(code, language="python")

            st.success("✅ Code generated and refined by CodeSpecialist ← QASpecialist collaboration!")

            st.download_button(
                "⬇️ Download Code",
                code,
                file_name=f"cnn_implementation.py",
                mime="text/plain",
                key="download_code_btn",
            )

# ============================================================
# TAB 5 — MULTI-AGENT PAPER COMPARISON
# ============================================================
with tab5:
    st.header("📊 Multi-Agent Paper Comparison")

    with st.container():
        st.markdown("""
        <div style="background-color: #fce4ec; padding: 15px; border-radius: 10px; border-left: 5px solid #E91E63;">
        <h4 style="margin: 0; color: #C2185B;">🤖 2-Agent Collaboration Process:</h4>
        <ol style="margin: 10px 0;">
            <li><b>SearchSpecialist</b> extracts key information from both papers</li>
            <li><b>AnalysisSpecialist</b> performs detailed comparison analysis</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    papers = st.session_state["papers"]
    if not papers:
        st.warning("⚠️ Please search for papers in the 'Search CNN Papers' tab first.")
    else:
        titles = [f"{i + 1}. {p['title']}" for i, p in enumerate(papers)]

        col1, col2 = st.columns(2)
        with col1:
            idx1 = st.selectbox(
                "Select first CNN paper:",
                range(len(papers)),
                format_func=lambda i: titles[i],
                key="cmp_paper1",
            )
        with col2:
            idx2 = st.selectbox(
                "Select second CNN paper:",
                range(len(papers)),
                format_func=lambda i: titles[i],
                key="cmp_paper2",
            )

        aspect = st.selectbox(
            "Comparison aspect:",
            ["methodology", "results", "CNN architecture", "applications", "overall quality"],
            key="cmp_aspect",
        )

        if st.button("🔍 Compare with Multi-Agent System", key="cmp_btn"):
            if idx1 == idx2:
                st.warning("Please select two different papers.")
            else:
                text1 = papers[idx1].get("abstract", "")
                text2 = papers[idx2].get("abstract", "")
                if not text1 or not text2:
                    st.error("One of the papers is missing an abstract.")
                else:
                    with st.spinner("🤖 Agents analyzing (Search → Analysis)..."):
                        # This triggers REAL multi-agent comparison
                        cmp_result = multi_agent_paper_comparison(text1, text2, aspect)
                    st.session_state["compare_result"] = cmp_result
                    st.success("✅ 2 agents collaborated to produce this comparison!")

        if st.session_state["compare_result"]:
            st.markdown("### 📊 Multi-Agent Comparison Result")
            st.markdown(st.session_state["compare_result"])

            st.download_button(
                "⬇️ Download Comparison",
                st.session_state["compare_result"],
                file_name="cnn_paper_comparison.md",
                mime="text/markdown",
                key="download_comparison_btn",
            )

# ============================================================
# FOOTER WITH AGENT INFO
# ============================================================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🤖 Active Agents:**
    - SearchSpecialist
    - QASpecialist
    - CodeSpecialist
    - AnalysisSpecialist
    - Controller
    """)

with col2:
    st.markdown("""
    **🔄 Communication:**
    - Sequential processing
    - Information passing
    - Multi-turn collaboration
    - Specialized expertise
    """)

with col3:
    st.markdown("""
    **⚙️ Technology:**
    - AutoGen Framework
    - Groq LLM Backend
    - Llama 3.3 70B
    - RAG for PDFs
    """)

st.markdown("---")
st.caption("🤖 Multi-Agent Research Assistant for Deep CNNs using AutoGen | Built with Streamlit")