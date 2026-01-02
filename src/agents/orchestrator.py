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
        # Using GPT-4o-mini as the master agent for efficiency
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            api_key=api_key,
            tags=["orchestrator"]
        )
        
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
                description="""Use for queries requiring statistical data, numbers, or lists of jobs from the SQL database. 
                Examples: 'How many Python vacancies are there?', 'Show 5 Data Science jobs'."""
            ),
            Tool(
                name="rag_career_advice",
                func=self.rag_agent.run,
                description="""Use for descriptive queries about job qualification details, career advice, company information, 
                or general career knowledge from documents."""
            )
        ]
        
        # 2. Definisikan System Prompt
        system_prompt = """You are a Master AI Career Advisor. Your goal is to help users with their career queries by using the appropriate tools.
            
            [CRITICAL LANGUAGE CONSTRAINT]
            - Detect the language of the user's latest query (English or Indonesian).
            - You MUST provide your FINAL response in that SAME language.
            - If tools return information in Indonesian but the user asked in English, you MUST translate the findings to English.
            - If tools return information in English but the user asked in Indonesian, you MUST translate the findings to Indonesian.
            - NEVER switch to Indonesian if the user is asking in English, even if the job data is in Indonesian.

            GUIDELINES:
            1. Use 'sql_job_stats' for quantitative data (counts, lists, comparisons of numbers).
            2. Use 'rag_career_advice' for qualitative info (qualifications, advice, descriptions).
            3. You can use BOTH tools sequentially if a query requires it (e.g., 'How many Python jobs are there and what skills do they need?').
            4. If the user is just greeting or talking casually, respond politely without using tools.
            5. Do NOT provide intermediate responses or summaries after each tool call.
            6. Gather ALL necessary information from tools first, THEN provide ONE comprehensive final response in the user's language.
            7. Be professional, encouraging, and helpful."""
            
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
                elif hasattr(m, "content"): # Already a message object
                    converted.append(m)
        elif isinstance(chat_history, str):
            # If string, treat as initial context
            converted.append(HumanMessage(content=f"Previous conversation context:\n{chat_history}"))
            
        return converted

    def route_request(self, user_query, history_text):
        """
        Legacy support for advisor chat context.
        """
        return self.route_query(user_query, chat_history=history_text)

    def route_query(self, user_query: str, chat_history: any = None) -> str:
        """
        Main entry point untuk memproses query menggunakan API langchain 1.0+.
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
            return f"Sorry, there was a technical issue: {str(e)}"

    def stream_query(self, user_query: str, chat_history: any = None):
        """
        Streaming version of route_query with deep transparency and sub-agent tracking.
        """
        import time
        from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
        
        start_time = time.perf_counter()
        total_input_tokens = 0
        total_output_tokens = 0
        current_agent = "orchestrator"
        
        try:
            formatted_history = self._convert_history(chat_history)
            messages = formatted_history + [HumanMessage(content=user_query)]
            
            logger.info(f"Master Agent deep streaming query: {user_query}")
            
            # Use multi-mode stream for maximum detail. subgraphs=True yields (path, mode, data).
            for _, mode, data in self.agent.stream(
                {"messages": messages},
                stream_mode=["updates", "messages", "custom"],
                config={"callbacks": [self.langfuse_handler]},
                subgraphs=True
            ):
                if mode == "updates":
                    for node_name, state in data.items():
                        if "messages" in state:
                            msg = state["messages"][-1]
                            
                            # Detect Tool Usage
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    yield "thought", f"🛠️ **Using tool:** `{tc['name']}`"
                            elif isinstance(msg, ToolMessage):
                                content_snippet = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                                yield "thought", f"✅ **Tool finished.** Output: \n```\n{content_snippet}\n```"

                elif mode == "custom":
                    if isinstance(data, dict):
                        event_type = data.get("type")
                        content = data.get("content")
                        if event_type == "sql_query":
                            yield "thought", f"🔍 **Generating SQL:**\n```sql\n{content}\n```"
                        elif event_type == "rag_search":
                            yield "thought", f"📖 **Searching Knowledge Base for:** `{content}`"

                elif mode == "messages":
                    token, metadata = data
                    tags = metadata.get("tags", [])
                    
                    # Update Usage Metadata
                    if hasattr(token, "usage_metadata") and token.usage_metadata:
                        total_input_tokens = token.usage_metadata.get("input_tokens", total_input_tokens)
                        total_output_tokens = token.usage_metadata.get("output_tokens", total_output_tokens)
                    
                    # Track which agent is speaking
                    if tags:
                        this_agent = tags[0]
                        if this_agent != current_agent and this_agent in ["sql_agent", "rag_agent"]:
                            yield "thought", f"🤖 **{this_agent.replace('_', ' ').title()}** starts processing..."
                            current_agent = this_agent
                        elif "orchestrator" in tags:
                            current_agent = "orchestrator"
                    if hasattr(token, "content") and token.content:
                        # Output ONLY content tagged with 'orchestrator' to hide internal agent dialogue
                        if tags and "orchestrator" in tags:
                            yield "content", token.content

            # Final Metadata
            yield "metadata", {
                "latency": time.perf_counter() - start_time,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens
            }

        except Exception as e:
            logger.error(f"Orchestrator Deep Stream Error: {str(e)}")
            yield "content", f"Sorry, there was a technical issue during streaming: {str(e)}"

if __name__ == "__main__":
    orchestrator = Orchestrator()
    # Test simple query
    print(orchestrator.route_query("Halo, apa kabar?"))
