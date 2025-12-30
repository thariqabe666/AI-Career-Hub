from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
import os
import logging
from dotenv import load_dotenv

# Konfigurasi Logging agar kita bisa lihat error di Streamlit Cloud Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class SQLAgent:
    def __init__(self, db_path: str = None):
        # 1. Dapatkan Path Absolut dari Root Project
        # file ini ada di: src/agents/sql_agent.py
        current_file_path = os.path.abspath(__file__)
        agents_dir = os.path.dirname(current_file_path) # src/agents
        src_dir = os.path.dirname(agents_dir) # src
        project_root = os.path.dirname(src_dir) # root
        
        if db_path is None:
            # Mengarah ke: root/data/processed/jobs.db
            db_path = os.path.join(project_root, 'data', 'processed', 'jobs.db')
        
        # 2. Normalisasi path untuk Linux (Streamlit Cloud)
        db_path = os.path.abspath(db_path)
        
        # DEBUG: Muncul di log Streamlit Cloud untuk memastikan path benar
        logger.info(f"Mencoba mengakses database di: {db_path}")

        # 3. Cek apakah file benar-benar ada sebelum koneksi
        if not os.path.exists(db_path):
            logger.error("FILE DATABASE TIDAK DITEMUKAN!")
            # Tampilkan list file di folder tersebut untuk memudahkan debug
            parent_dir = os.path.dirname(db_path)
            if os.path.exists(parent_dir):
                logger.info(f"Isi folder {parent_dir}: {os.listdir(parent_dir)}")
            else:
                logger.error(f"Folder {parent_dir} juga tidak ditemukan!")
            
            raise FileNotFoundError(f"Database tidak ditemukan di {db_path}")

        # 4. Gunakan URI yang benar
        db_uri = f"sqlite:///{db_path}"
        self.db = SQLDatabase.from_uri(db_uri)
        
        # 5. Initialize Toolkit & LLM
        from langgraph.config import get_stream_writer
        
        from langfuse.langchain import CallbackHandler
        self.langfuse_handler = CallbackHandler()
        
        self.llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4o-mini", 
            api_key=os.getenv("OPENAI_API_KEY"),
            tags=["sql_agent"]
        )
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Wrap tools to emit custom events for streaming transparency
        original_tools = self.toolkit.get_tools()
        self.tools = []
        for t in original_tools:
            if t.name == "sql_db_query":
                def create_wrapped_tool(original_tool):
                    def wrapped_query(query: str, **kwargs):
                        writer = get_stream_writer()
                        # Emit a custom event with the SQL query
                        writer({"type": "sql_query", "content": query})
                        # Use .run() instead of .func because it might be a class-based tool
                        return original_tool.run(query, **kwargs)
                    return wrapped_query
                
                # Clone tool and replace func
                from langchain_core.tools import Tool
                new_tool = Tool(
                    name=t.name,
                    func=create_wrapped_tool(t),
                    description=t.description
                )
                self.tools.append(new_tool)
            else:
                self.tools.append(t)
        
        # 6. Define System Prompt (based on latest docs)
        system_prompt = """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        Then you should query the schema of the most relevant tables.
        """.format(
            dialect=self.db.dialect,
            top_k=5,
        )

        # 7. Create Agent using create_agent (latest pattern)
        from langchain.agents import create_agent
        self.agent_executor = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

    def run(self, query: str) -> str:
        """Standard execution (legacy/sync)"""
        try:
            # Re-implementing run using the new agent_executor (which is a CompiledStateGraph)
            response = self.agent_executor.invoke(
                {"messages": [("user", query)]},
                config={"callbacks": [self.langfuse_handler]}
            )
            return response["messages"][-1].content
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return f"Error database: {str(e)}"

    def stream(self, query: str):
        """Streaming execution for granular updates"""
        return self.agent_executor.stream(
            {"messages": [("user", query)]},
            stream_mode=["updates", "messages"],
            config={"callbacks": [self.langfuse_handler]}
        )

if __name__ == "__main__":
    agent = SQLAgent()
    print(agent.run("Berapa jumlah data di database?"))
