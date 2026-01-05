import streamlit as st
import os
from dotenv import load_dotenv
from src.agents.orchestrator import Orchestrator
from src.agents.advisor_agent import AdvisorAgent
from src.agents.cover_letter_agent import CoverLetterAgent
from src.agents.interview_agent import InterviewAgent
from streamlit_mic_recorder import mic_recorder
import openai
import hashlib

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(page_title="AI Career Hub", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    /* Global Base */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Titles & Headers */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        color: #ffffff !important;
    }

    .gradient-text {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        line-height: 1.2;
        text-align: center;
        width: 100%;
        display: block;
    }

    /* Glass Containers */
    .glass-container {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin: 1rem 0;
    }

    /* Job Cards */
    .job-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        height: 100%;
    }

    .job-card:hover {
        background: rgba(255, 255, 255, 0.06);
        transform: translateY(-8px) scale(1.02);
        border-color: #6366f1;
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.3);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        text-transform: none !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    }

    /* Form Inputs */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 0px;
        font-weight: 600;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        color: #6366f1 !important;
        border-bottom-color: #6366f1 !important;
    }

    /* Streamlit Overrides */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi Agent
@st.cache_resource
def init_agents():
    return {
        "orchestrator": Orchestrator(),
        "advisor": AdvisorAgent(),
        "cover_letter": CoverLetterAgent(),
        "interview": InterviewAgent()
    }

agents = init_agents()
 
def get_full_job_data(doc, agents):
    """
    Merges data from RAG (Qdrant) and SQL (SQLite) to provide complete job info.
    """
    data = doc.metadata.copy()
    
    # Description is often in page_content (RAG)
    if 'page_content' in data:
        data['description'] = data['page_content']
    elif hasattr(doc, 'page_content'):
        data['description'] = doc.page_content
        
    # Clean up title and company if keys are different
    if 'job_title' in data and 'title' not in data: data['title'] = data['job_title']
    if 'company_name' in data and 'company' not in data: data['company'] = data['company_name']
    
    # Try to fetch more details from SQL if sql_id exists
    sql_id = data.get('sql_id')
    if sql_id is not None:
        try:
            db = agents["orchestrator"].sql_agent.db
            sql_query = f"SELECT * FROM jobs_table WHERE id = {sql_id}"
            res = db.run(sql_query)
            # res is usually a string representation of list of tuples from langchain SQLDatabase
            # Example: "[(0, 'Title', 'Company', 'Location', 'Type', None, None)]"
            import ast
            res_list = ast.literal_eval(res)
            if res_list and len(res_list) > 0:
                row = res_list[0]
                # Mapping: (id, job_title, company_name, clean_location, work_type, min_salary, max_salary)
                data['location'] = row[3] if len(row) > 3 else data.get('location')
                data['type'] = row[4] if len(row) > 4 else data.get('type')
                
                min_s = row[5] if len(row) > 5 else None
                max_s = row[6] if len(row) > 6 else None
                if min_s or max_s:
                    data['salary'] = f"{min_s or '?'}-{max_s or '?'}"
                else:
                    data['salary'] = data.get('salary', 'Competitive')
        except Exception as e:
            print(f"Error fetching SQL data: {e}")
            
    # Final normalization
    return {
        "title": data.get("title", "Unknown Title"),
        "company": data.get("company", "Unknown Company"),
        "location": data.get("location", "Remote/Not specified"),
        "description": data.get("description", "No description available..."),
        "type": data.get("type", "Not specified"),
        "salary": data.get("salary", "Competitive"),
        "sql_id": sql_id
    }

# Initialize Session State
if "track" not in st.session_state:
    st.session_state.track = "🚀 Career Co-Pilot"
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "cv_text" not in st.session_state:
    st.session_state.cv_text = None
if "jobs_list" not in st.session_state:
    st.session_state.jobs_list = []
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "advisor_report" not in st.session_state:
    st.session_state.advisor_report = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_session():
    st.session_state.page = "landing"
    st.session_state.cv_text = None
    st.session_state.jobs_list = []
    st.session_state.selected_job = None
    st.session_state.advisor_report = None
    st.session_state.messages = []
    if "advisor_messages" in st.session_state: del st.session_state.advisor_messages
    if "interview_log" in st.session_state: del st.session_state.interview_log
    st.rerun()

# Sidebar Navigation
with st.sidebar:
    st.title("🛡️ AI Career Hub")
    st.markdown("---")
    selected_track = st.radio("Navigation", ["🚀 Career Co-Pilot", "💬 Smart Chat"])
    
    if selected_track != st.session_state.track:
        st.session_state.track = selected_track
        st.rerun()
        
    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        reset_session()

# --- TRACK 1: CAREER CO-PILOT ---
if st.session_state.track == "🚀 Career Co-Pilot":
    # --- PHASE 1: LANDING PAGE ---
    if st.session_state.page == "landing":
        # Hero Section
        st.markdown('<div style="text-align: center; padding: 5rem 0 2rem 0;">', unsafe_allow_html=True)
        st.markdown('<h1 class="gradient-text">Unlock Your Future</h1>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 1.25rem; color: #94a3b8; max-width: 700px; margin: 0 auto 3rem auto; text-align: center; line-height: 1.6;">Upload your CV and let our AI agents guide you to your dream career. We analyze, recommend, and prepare you for success.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Upload Section
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<p style="text-align: center; color: #6366f1; font-weight: 600; margin-bottom: 0.5rem;">READY TO START?</p>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Drop your CV here (PDF)", type=["pdf"], label_visibility="collapsed")
            if uploaded_file:
                st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
                if st.button("Find My Path 🚀", use_container_width=True):
                    with st.status("Analyzing your profile...", expanded=True) as status:
                        # Save temp file
                        temp_path = "temp_cv_landing.pdf"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extract Text & Profile
                        st.write("Extracting skills and experience...")
                        cv_text = agents["advisor"].extract_text_from_pdf(temp_path)
                        st.session_state.cv_text = cv_text
                        
                        # Get Search Query for RAG
                        st.write("Identifying relevant job markets...")
                        profile_prompt = "Analyze this CV and extract core skills, experience level, and preferred roles. Output a search query for job lookup.\n\nCV Content:\n" + cv_text[:5000]
                        search_query = agents["advisor"].llm.invoke(profile_prompt).content
                        
                        # Retrieve Jobs
                        st.write("Searching for matching opportunities...")
                        job_docs = agents["advisor"].rag_agent.retrieve_documents(search_query, limit=6)
                        st.session_state.jobs_list = [get_full_job_data(doc, agents) for doc in job_docs]
                        
                        # Initial Consultation Report
                        st.write("Generating your career roadmap...")
                        st.session_state.advisor_report = agents["advisor"].analyze_and_recommend(temp_path)
                        
                        os.remove(temp_path)
                        status.update(label="Discovery Complete!", state="complete", expanded=False)
                    
                    st.session_state.page = "dashboard"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # --- PHASE 2: DASHBOARD ---
    elif st.session_state.page == "dashboard":
        st.header("🎯 Recommended Opportunities")
        st.write("Based on your profile, we've identified these high-potential matches:")
        
        if not st.session_state.jobs_list:
            st.warning("No jobs found matching your profile. Try resetting and uploading again.")
            if st.button("Back to Home"):
                st.session_state.page = "landing"
                st.rerun()
        else:
            # Create a Grid
            cols = st.columns(3)
            for i, job in enumerate(st.session_state.jobs_list):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="job-card">
                        <h3>{job.get('title', 'Unknown Title')}</h3>
                        <p style="color: #6366f1; font-weight: 600;">{job.get('company', 'Unknown Company')}</p>
                        <p style="font-size: 0.9rem; color: #94a3b8;">📍 {job.get('location', 'Remote')}</p>
                        <p style="font-size: 0.85rem; line-height: 1.4; height: 3em; overflow: hidden;">{job.get('description', 'No description available...')[:100]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Select: {job.get('title')[:20]}", key=f"job_{i}", use_container_width=True):
                        st.session_state.selected_job = job
                        st.session_state.page = "workspace"
                        st.rerun()
            
            # --- CUSTOM JOB INPUT ---
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h3 style="margin-bottom: 0.5rem;">Can't find the right role? 🔍</h3>
                <p style="color: #94a3b8;">Input a specific job description to get tailored analysis and interview practice.</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("✨ Process a Custom Job Description", expanded=False):
                with st.form("custom_job_form"):
                    cust_title = st.text_input("Job Title", placeholder="e.g. Senior Backend Engineer")
                    cust_company = st.text_input("Company Name", placeholder="e.g. Google")
                    cust_desc = st.text_area("Job Description", placeholder="Paste the full job description here...", height=200)
                    
                    submit_custom = st.form_submit_button("Launch Workspace 🚀", use_container_width=True)
                    
                    if submit_custom:
                        if not cust_title or not cust_desc:
                            st.error("Please provide at least a Job Title and Description.")
                        else:
                            st.session_state.selected_job = {
                                "title": cust_title,
                                "company": cust_company if cust_company else "Custom Company",
                                "description": cust_desc,
                                "location": "N/A",
                                "type": "N/A",
                                "salary": "N/A"
                            }
                            st.session_state.page = "workspace"
                            # Reset job-specific state just in case
                            for key in ["interview_log", "interview_history_text", "current_q", "question_count", "interview_ended", "evaluation_report", "max_questions"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()

    # --- PHASE 3: WORKSPACE ---
    elif st.session_state.page == "workspace":
        job = st.session_state.selected_job
        
        # Header with Back Button
        col_back, col_title = st.columns([1, 5])
        with col_back:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.page = "dashboard"
                # Reset job-specific session state
                keys_to_reset = [
                    "selected_job", "interview_log", "interview_history_text", 
                    "current_q", "question_count", "interview_ended", 
                    "evaluation_report", "max_questions"
                ]
                for key in keys_to_reset:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        with col_title:
            st.markdown(f"## 🛠️ Workspace: <span style='color: #818cf8;'>{job.get('title')}</span> @ {job.get('company')}", unsafe_allow_html=True)
        
        # Display Job Info in a nice glass container
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"""
                <div class="glass-container" style="padding: 1.5rem; margin-bottom: 1rem;">
                    <h4 style="margin-top:0;">Job Details</h4>
                    <p><b>📍 Location:</b> {job.get('location', 'Not specified')}</p>
                    <p><b>💼 Type:</b> {job.get('type', 'Not specified')}</p>
                    <p><b>💰 Salary:</b> {job.get('salary', 'Competitive')}</p>
                    <p><b>🏢 Company:</b> {job.get('company', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="glass-container" style="padding: 1.5rem; margin-bottom: 1rem;">
                    <h4 style="margin-top:0;">Description</h4>
                    <p style="font-size: 0.9rem; max-height: 120px; overflow-y: auto;">{job.get('description', 'No description available.')}</p>
                </div>
                """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Match Analysis", "📝 Cover Letter", "🎙️ Interview Sim"])
        
        with tab1:
            st.subheader("Match Analysis")
            if st.button("Start Deep Analysis"):
                with st.spinner("Analyzing compatibility..."):
                    results = agents["advisor"].get_match_analysis(
                        st.session_state.cv_text, 
                        job.get('description')
                    )
                    
                    # Display Score
                    score = results.get("match_score", 0)
                    st.markdown(f"### Match Score: {score}%")
                    st.progress(score / 100)
                    
                    st.markdown(f"**Summary:** {results.get('summary', '')}")
                    
                    col_s, col_g = st.columns(2)
                    with col_s:
                        st.success("✅ **Key Strengths**")
                        for s in results.get("strengths", []):
                            st.write(f"- {s}")
                    with col_g:
                        st.error("⚠️ **Skill Gaps**")
                        for g in results.get("gaps", []):
                            st.write(f"- {g}")
                    
                    st.info("💡 **Actionable Recommendations**")
                    for r in results.get("recommendations", []):
                        st.write(f"- {r}")
            else:
                st.info("Click the button above to start a deep compatibility analysis between your CV and this job.")

        with tab2:
            st.subheader("Tailored Cover Letter")
            if st.button("Generate My Letter"):
                with st.spinner("Writing a winning cover letter..."):
                    prompt = f"""
                    Write a cover letter for {job.get('title')} at {job.get('company')}.
                    DESCRIPTION: {job.get('description')}
                    CANDIDATE INFO: {st.session_state.cv_text[:3000]}
                    """
                    letter = agents["cover_letter"].llm.invoke(prompt).content
                    st.text_area("Copy your letter:", value=letter, height=400)
            else:
                st.info("Let the AI write a cover letter that highlights your strengths.")

        with tab3:
            st.subheader("Interactive Interview Simulator")
            
            # --- Initialize Interview State ---
            if "interview_log" not in st.session_state:
                st.session_state.interview_log = []
            if "interview_history_text" not in st.session_state:
                st.session_state.interview_history_text = f"You are interviewing for {job.get('title')} at {job.get('company')}.\n"
                st.session_state.current_q = f"Hello! Let's start the interview for the {job.get('title')} position. Could you introduce yourself?"
                st.session_state.interview_log.append({"role": "assistant", "content": st.session_state.current_q})
                st.session_state.question_count = 1
                st.session_state.interview_ended = False
                st.session_state.evaluation_report = None

            # --- Layout: Main Chat vs Sidebar Tips ---
            col_chat, col_info = st.columns([2, 1])

            with col_info:
                st.markdown(f"""
                <div class="glass-container" style="padding: 1.5rem;">
                    <h4 style="margin-top:0;">Session Info</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Input for question limit - only show before the interview starts or in sidebar
                if "max_questions" not in st.session_state:
                    st.session_state.max_questions = 5
                
                if st.session_state.question_count == 1 and not st.session_state.interview_ended:
                    st.session_state.max_questions = st.number_input(
                        "Number of questions to practice:", 
                        min_value=1, max_value=10, value=5, step=1
                    )
                else:
                    st.write(f"**Target Questions:** {st.session_state.max_questions}")
                
                st.write(f"**Current Progress:** {st.session_state.question_count} / {st.session_state.max_questions}")
                st.markdown('<p style="font-size: 0.9rem; color: #94a3b8;">The interview will automatically conclude after reaching the target for a full evaluation.</p>', unsafe_allow_html=True)
                
                if not st.session_state.interview_ended:
                    if st.button("🏁 End Interview Early", use_container_width=True):
                        st.session_state.interview_ended = True
                        st.rerun()
                
                if st.button("🔄 Reset Session", key="reset_int", use_container_width=True):
                    del st.session_state.interview_log
                    del st.session_state.interview_history_text
                    if "question_count" in st.session_state: del st.session_state.question_count
                    if "interview_ended" in st.session_state: del st.session_state.interview_ended
                    if "evaluation_report" in st.session_state: del st.session_state.evaluation_report
                    if "max_questions" in st.session_state: del st.session_state.max_questions
                    st.rerun()

            with col_chat:
                if not st.session_state.interview_ended:
                    # Message from typical chat interface
                    st.markdown(f"**Interviewer:** {st.session_state.current_q}")
                    
                    audio_data = mic_recorder(
                        start_prompt="Click to Speak 🎤", 
                        stop_prompt="Stop & Send ✅", 
                        key='workspace_mic'
                    )
                    
                    if audio_data:
                        audio_bytes = audio_data['bytes']
                        audio_hash = hashlib.md5(audio_bytes).hexdigest()
                        
                        if "last_processed_audio_hash" not in st.session_state or st.session_state.last_processed_audio_hash != audio_hash:
                            with st.status("Analyzing your response...", expanded=False):
                                # 1. Transcription
                                with open("temp_ws_int.mp3", "wb") as f: f.write(audio_bytes)
                                client = openai.OpenAI()
                                with open("temp_ws_int.mp3", "rb") as audio_file:
                                    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                                user_text = transcript.text
                                st.session_state.interview_log.append({"role": "user", "content": user_text})
                                
                                # 2. Check for End of Session
                                if st.session_state.question_count >= st.session_state.max_questions:
                                    st.session_state.interview_ended = True
                                else:
                                    # 3. Get LLM Response
                                    response = agents["interview"].get_response(
                                        st.session_state.interview_history_text, 
                                        user_text,
                                        job.get('description', ''),
                                        st.session_state.cv_text or ''
                                    )
                                    st.session_state.interview_log.append({"role": "assistant", "content": response})
                                    st.session_state.interview_history_text += f"Candidate: {user_text}\nInterviewer: {response}\n"
                                    st.session_state.current_q = response
                                    st.session_state.question_count += 1
                                
                                st.session_state.last_processed_audio_hash = audio_hash
                                os.remove("temp_ws_int.mp3")
                            st.rerun()
                else:
                    # --- EVALUATION PHASE ---
                    if not st.session_state.evaluation_report:
                        with st.status("Generating Final Evaluation...", expanded=True):
                            st.write("Reviewing interview history...")
                            report = agents["interview"].evaluate_session(
                                st.session_state.interview_history_text,
                                job.get('description', ''),
                                st.session_state.cv_text or ''
                            )
                            # Clean up potential code block wrapping
                            if report.startswith("```markdown"):
                                report = report[11:]
                            if report.startswith("```"):
                                report = report[3:]
                            if report.endswith("```"):
                                report = report[:-3]
                            
                            st.session_state.evaluation_report = report.strip()
                            st.rerun()
                    
                    st.success("🎉 Interview Complete!")
                    
                    # --- Improved Score UI ---
                    report_text = st.session_state.evaluation_report
                    score_val = "N/A"
                    remaining_report = report_text
                    
                    # Try to extract score from "# 🏆 OVERALL SCORE: [value]"
                    import re
                    score_match = re.search(r"# 🏆 OVERALL SCORE:\s*(\d+)", report_text)
                    if score_match:
                        score_val = score_match.group(1)
                        # Remove the score line from the report to avoid duplication
                        remaining_report = re.sub(r"# 🏆 OVERALL SCORE:\s*\d+", "", report_text).strip()
                    
                    col_score, col_empty = st.columns([1, 2])
                    with col_score:
                        st.metric("Overall Performance", f"{score_val}/100")
                    
                    st.markdown("---")
                    # Render the remaining evaluation as clean Markdown
                    st.markdown(remaining_report)

                # --- Conversation History ---
                st.markdown("---")
                st.markdown("### 💬 Conversation History")
                for msg in reversed(st.session_state.interview_log):
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

# --- TRACK 2: SMART CHAT ---
elif st.session_state.track == "💬 Smart Chat":
    st.markdown('<h2 class="gradient-text" style="font-size: 2rem;">Smart Career Assistant</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.25rem; color: #94a3b8; max-width: 700px; margin: 0 auto 3rem auto; text-align: center; line-height: 1.6;">Explore job market statistics or get detailed information about roles.</p>', unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about job trends, requirements, or salaries..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.status("Searching knowledge base & statistics...", expanded=True)
            
            def stream_handler():
                full_response = ""
                for mode, content in agents["orchestrator"].stream_query(prompt, chat_history=st.session_state.messages):
                    if mode == "thought":
                        status.write(content)
                    elif mode == "content":
                        full_response += content
                        yield content
                    elif mode == "metadata":
                        st.session_state.last_metadata = content
                status.update(label="Response generated", state="complete", expanded=False)

            response = st.write_stream(stream_handler())
            
            if "last_metadata" in st.session_state:
                m = st.session_state.last_metadata
                st.caption(f"⚡ {m['latency']:.1f}s | 📥 {m['input_tokens']} tokens | 📤 {m['output_tokens']} tokens")
                del st.session_state.last_metadata
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
