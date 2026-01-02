import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import logging
from typing import List, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from src.database.setup_qdrant import get_qdrant_client
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain_core.callbacks import StdOutCallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RAGAgent:
    def __init__(self, collection_name: str = "job_market"):
        """
        Initializes the RAG Agent with a Qdrant client, Embedding model, and LLM.
        """
        self.collection_name = collection_name
        self.client = get_qdrant_client()
        
        api_key = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            api_key=api_key,
            tags=["rag_agent"]
        )
        
        # Initialize Langfuse CallbackHandler
        self.langfuse_handler = CallbackHandler()

        # Define the search tool for the agent
        @tool
        def search_knowledge_base(query: str) -> str:
            """Searches the career knowledge base for relevant documents and information."""
            try:
                from langgraph.config import get_stream_writer
                writer = get_stream_writer()
                if writer:
                    writer({"type": "rag_search", "content": query})
            except Exception:
                pass
            
            docs = self.retrieve_documents(query)
            if not docs:
                return "No specific data found in the knowledge base."
            return "\n\n".join([doc.page_content for doc in docs])

        self.tools = [search_knowledge_base]
        
        system_prompt = """You are a professional Career Assistant. 
        Your task is to answer user questions using the 'search_knowledge_base' tool.
        
        INSTRUCTIONS:
        1. LANGUAGE: ALWAYS respond in the SAME LANGUAGE as the user's latest query. If retrieved info is in a different language, translate it.
        2. SMART SEARCH: Use the tool to find relevant data.
        3. FALLBACK: If the tool returns no specific data, provide a high-quality response based on your general career knowledge.
        4. TONE: Maintain a friendly, professional, and encouraging persona.
        
        Respond clearly based on the retrieved information or your general expertise."""

        # Create a ReAct-style agent to show thinking steps
        from langchain.agents import create_agent
        self.agent_executor = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

    def retrieve_documents(self, query: str, limit: int = 3) -> List[Document]:
        """
        Embeds the query and searches the Qdrant collection.
        Returns a list of LangChain Documents.
        """
        try:
            query_vector = self.embeddings.embed_query(query)
            
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            ).points
            
            documents = []
            for hit in search_results:
                page_content = hit.payload.get("text", hit.payload.get("content", str(hit.payload)))
                metadata = hit.payload
                documents.append(Document(page_content=page_content, metadata=metadata))
            
            return documents
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def run(self, query: str) -> str:
        """
        End-to-end RAG run using an Agent to show thinking steps.
        """
        logger.info(f"RAG Agent received query: {query}")
        try:
            response = self.agent_executor.invoke(
                {"messages": [("user", query)]},
                config={"callbacks": [self.langfuse_handler]}
            )
            return response["messages"][-1].content
        except Exception as e:
            logger.error(f"Error in RAG Agent: {e}")
            return f"Sorry, there was a technical issue while searching: {str(e)}"

if __name__ == "__main__":
    agent = RAGAgent()
    print(agent.run("What are the soft skills for Python developers?"))
