import os
import sys
import logging
import pypdf
from pypdf import PdfReader

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.agents import AdvisorAgent, RAGAgent, SQLAgent, Orchestrator
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("--- Starting Sandbox Test ---")

    # 1. Test SQL Agent (if available)
    if SQLAgent:
        print("\n--- Testing SQL Agent Initialization ---")
        try:
            sql_agent = SQLAgent()
            print("SQL Agent initialized successfully.")
            # detailed test could be added here if we knew the schema/query
        except Exception as e:
            print(f"Failed to initialize SQL Agent: {e}")

    # 2. Test RAG Agent (standalone)
    print("\n--- Testing RAG Agent Initialization ---")
    try:
        rag_agent = RAGAgent()
        print("RAG Agent initialized successfully.")
        # Test a simple retrieval
        docs = rag_agent.retrieve_documents("AI Engineer", limit=1)
        print(f"RAG Retrieval Test: Found {len(docs)} documents.")
    except Exception as e:
        print(f"Failed to initialize or run RAG Agent: {e}")

    # 3. Test Advisor Agent with CV
    print("\n--- Testing Advisor Agent with CV ---")
    cv_path = r"c:\Users\Thariq Ahmad Baihaqi\Documents\#AI ENGINEERING\LATIHAN KODING PURRRWADHIKA\#FinalProj\Thariq Ahmad B.A. - AI Engineering - CV.pdf"
    
    if not os.path.exists(cv_path):
        print(f"Error: CV file not found at {cv_path}")
        return

    try:
        advisor = AdvisorAgent()
        print("Advisor Agent initialized successfully.")
        
        print(f"Running analyze_and_recommend on {cv_path}...")
        recommendation = advisor.analyze_and_recommend(cv_path)
        
        print("\n=== Career Consultation Recommendation ===")
        print(recommendation)
        print("==========================================")
        
    except Exception as e:
        print(f"Error running Advisor Agent: {e}")
        import traceback
        traceback.print_exc()

    # 4. Test Orchestrator
    print("\n--- Testing Orchestrator ---")
    try:
        orchestrator = Orchestrator()
        print("Orchestrator initialized successfully.")
        
        q1 = "How many jobs are there?"
        print(f"Testing Query: '{q1}'")
        res1 = orchestrator.route_query(q1)
        print(f"Result: {res1}")
        
        q2 = "Find me a python developer job."
        print(f"Testing Query: '{q2}'")
        res2 = orchestrator.route_query(q2)
        print(f"Result: {res2}")
        
    except Exception as e:
        print(f"Error running Orchestrator: {e}")


if __name__ == "__main__":
    main()
