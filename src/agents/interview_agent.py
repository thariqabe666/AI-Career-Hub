import os
import speech_recognition as sr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class InterviewAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        # Prompt yang mewajibkan AI menjawab sesuai bahasa user
        self.prompt = ChatPromptTemplate.from_template("""
            You are a professional Interviewer. 
            
            JOB DESCRIPTION:
            {job_description}
            
            CANDIDATE CV:
            {cv_text}
            
            HISTORY: {history}
            CANDIDATE ANSWER: {answer}
            
            INSTRUCTIONS:
            1. CRITICAL: Response in the EXACT SAME LANGUAGE as the candidate's last answer. If they speak English, you MUST speak English. If they speak Indonesian, you MUST speak Indonesian.
            2. Give brief feedback on the answer based on the job requirements and candidate's CV.
            3. Ask exactly ONE follow-up question that is relevant to the role and the candidate's previous answer.
            
            YOUR RESPONSE:
        """)

    def get_response(self, history, user_answer, job_description, cv_text):
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "history": history, 
            "answer": user_answer,
            "job_description": job_description,
            "cv_text": cv_text
        })

    def evaluate_session(self, history, job_description, cv_text):
        """
        Evaluates the entire interview session.
        """
        eval_prompt = ChatPromptTemplate.from_template("""
            You are an expert HR Interview Evaluator. 
            Analyze the following interview history between an Interviewer and a Candidate for a specific job.
            
            JOB DESCRIPTION:
            {job_description}
            
            CANDIDATE CV:
            {cv_text}
            
            INTERVIEW HISTORY:
            {history}
            
            INSTRUCTIONS:
            1. CRITICAL: You MUST evaluate the candidate in the EXACT SAME LANGUAGE they used primarily during the interview. 
            2. If the candidate spoke English, the entire evaluation MUST be in English. 
            3. If the candidate spoke Indonesian, the entire evaluation MUST be in Indonesian.
            4. DO NOT mix languages. DO NOT use Indonesian if the interview was in English.
            
            Output your evaluation in Markdown format with the EXACT following structure:
            
            # 🏆 OVERALL SCORE: [Insert Number 0-100 here]
            
            ## 📝 Session Summary
            [Briefly summarize how the interview went]
            
            ## ✅ Key Strengths
            - [Strength 1]
            - [Strength 2]
            - [Strength 3]
            
            ## ⚠️ Areas for Improvement
            - [Area 1]
            - [Area 2]
            - [Area 3]
            
            ## 💡 Actionable Insights
            [Advice for the candidate]
            
            YOUR EVALUATION:
        """)
        chain = eval_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "history": history,
            "job_description": job_description,
            "cv_text": cv_text
        })

    def listen(self):
        """
        Listens to the microphone and converts speech to text using OpenAI Whisper (via SpeechRecognition if available or API).
        For this implementation, we'll use local Google Speech Recognition as a default fallback, 
        or specifically configured for Whisper API if the user has keys/setup.
        
        To keep it simple and robust:
        We will use `recognizer.recognize_google` for quick testing if Whisper API isn't explicitly requested 
        but the plan mentioned "Send audio to OpenAI Whisper". 
        
        To use OpenAI Whisper API via speech_recognition:
        recognizer.recognize_whisper_api(audio, api_key=OPENAI_API_KEY)
        """
        with sr.Microphone() as source:
            print("\nListening... (Speak now)")
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=30)
                print("Processing audio...")
                
                # Using OpenAI Whisper API for better accuracy
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    text = self.recognizer.recognize_whisper_api(audio, api_key=api_key)
                else:
                    # Fallback to Google if no key (though this project seems to have one)
                    text = self.recognizer.recognize_google(audio)
                
                print(f"You said: {text}")
                return text
            except sr.WaitTimeoutError:
                print("No speech detected.")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio.")
                return None
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None

    def run(self):
        """
        Starts the interview loop.
        """
        print("Starting Mock Interview Agent...")
        print("Say 'exit' or 'stop' to end the interview.")
        
        # Initial greeting
        initial_question = "Tell me about yourself and your background in software engineering."
        print(f"\nAgent: {initial_question}")
        self.history += f"Agent: {initial_question}\n"
        
        while True:
            # 1. Listen
            user_input = self.listen()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "stop", "quit", "bye"]:
                print("Ending interview. Good luck!")
                break
            
            # 2. Update History
            self.history += f"Candidate: {user_input}\n"
            
            # 3. Generate Response
            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke({"history": self.history, "answer": user_input})
            
            # 4. Output Response
            print(f"\nAgent: {response}")
            
            # 5. Update History with Agent response
            self.history += f"Agent: {response}\n"
            

if __name__ == "__main__":
    agent = InterviewAgent()
    agent.run()
