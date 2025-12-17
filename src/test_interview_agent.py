import sys
import os

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.agents.interview_agent import InterviewAgent

def main():
    print("Initializing Interview Agent...")
    try:
        agent = InterviewAgent()
        print("Agent initialized successfully.")
        print("-" * 50)
        agent.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Ensure you have a microphone connected and dependencies installed listed in requirements.txt (PyAudio, SpeechRecognition).")

if __name__ == "__main__":
    main()
