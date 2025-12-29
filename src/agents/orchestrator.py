import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler

from .sql_agent import SQLAgent
from .rag_agent import RAGAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class Orchestrator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        # Menggunakan GPT-4o-mini untuk efisiensi sebagai master agent
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        
        # Inisialisasi sub-agents
        self.sql_agent = SQLAgent()
        self.rag_agent = RAGAgent()
        
        # Inisialisasi Langfuse CallbackHandler
        self.langfuse_handler = CallbackHandler()
        
        # 1. Definisikan Tools
        tools = [
            Tool(
                name="sql_job_stats",
                func=self.sql_agent.run,
                description="""Gunakan untuk pertanyaan yang membutuhkan data statistik, angka, atau daftar pekerjaan dari database SQL. 
                Contoh: 'Berapa jumlah lowongan Python?', 'Tampilkan 5 loker Data Science'."""
            ),
            Tool(
                name="rag_career_advice",
                func=self.rag_agent.run,
                description="""Gunakan untuk pertanyaan deskriptif tentang detail kualifikasi pekerjaan, saran karir, informasi perusahaan, 
                atau hal-hal yang bersifat pengetahuan umum karir dari dokumen PDF."""
            )
        ]
        
        # 2. Definisikan System Prompt
        system_prompt = """You are a Master AI Career Advisor. Your goal is to help users with their career queries by using the appropriate tools.
            
            GUIDELINES:
            1. Use 'sql_job_stats' for quantitative data (counts, lists, comparisons of numbers).
            2. Use 'rag_career_advice' for qualitative info (qualifications, advice, descriptions).
            3. You can use BOTH tools sequentially if a query requires it (e.g., 'How many Python jobs are there and what skills do they need?').
            4. If the user is just greeting or talking casually, respond politely without using tools.
            5. ALWAYS respond in the SAME LANGUAGE as the user (Indonesian or English).
            6. Be professional, encouraging, and helpful."""
            
        # 3. Inisialisasi Agent menggunakan API terbaru langchain 1.0+
        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt
        )

    def _convert_history(self, chat_history):
        """
        Mengonversi history dari berbagai format (list of dicts atau string) 
        ke list of LangChain Message objects.
        """
        converted = []
        if isinstance(chat_history, list):
            for m in chat_history:
                if isinstance(m, dict):
                    role = m.get("role")
                    content = m.get("content")
                    if role == "user":
                        converted.append(HumanMessage(content=content))
                    elif role == "assistant":
                        converted.append(AIMessage(content=content))
                elif hasattr(m, "content"): # Sudah objek message
                    converted.append(m)
        elif isinstance(chat_history, str):
            # Jika string, kita anggap sebagai satu konteks awal
            converted.append(HumanMessage(content=f"Konteks percakapan sebelumnya:\n{chat_history}"))
            
        return converted

    def route_request(self, user_query, history_text):
        """
        Legacy support for advisor chat context.
        """
        return self.route_query(user_query, chat_history=history_text)

    def route_query(self, user_query: str, chat_history: any = None) -> str:
        """
        Utama entry point untuk memproses query menggunakan API langchain 1.0+.
        """
        try:
            formatted_history = self._convert_history(chat_history)
            
            # Gabungkan sejarah dengan query saat ini
            messages = formatted_history + [HumanMessage(content=user_query)]
                
            logger.info(f"Master Agent processing query: {user_query}")
            
            # Invoke agent dengan format state yang diharapkan (messages)
            response = self.agent.invoke(
                {"messages": messages},
                config={"callbacks": [self.langfuse_handler]}
            )
            
            # Output akhir berada di pesan terakhir dari state messages
            return response["messages"][-1].content

        except Exception as e:
            logger.error(f"Orchestrator Error: {str(e)}")
            return f"Maaf, ada kendala teknis: {str(e)}"

if __name__ == "__main__":
    orchestrator = Orchestrator()
    # Test simple query
    print(orchestrator.route_query("Halo, apa kabar?"))
