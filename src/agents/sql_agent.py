from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

load_dotenv()

class SQLAgent:
    def __init__(self, db_uri: str = None):
        """
        Initialize the SQL Agent.
        
        Args:
            db_uri (str, optional): The database connection URI (e.g., 'sqlite:///titanic.db', 'postgresql://user:pass@localhost/dbname')
                                   If None, defaults to the project's jobs database.
        """
        # Use default database path if not provided
        if db_uri is None:
            # Get the project root directory (assuming sql_agent.py is in src/agents/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '../../'))
            db_path = os.path.join(project_root, 'data', 'processed', 'jobs.db')
            db_uri = f"sqlite:///{db_path}"
        
        self.db = SQLDatabase.from_uri(db_uri)
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Initialize Langfuse CallbackHandler
        self.langfuse_handler = CallbackHandler()

    def get_schema_info(self):
        """
        Returns the schema information of the database.
        """
        return self.db.get_table_info()

    def run(self, query: str) -> str:
        """
        Run the agent with the given query.
        
        Args:
            query (str): The natural language query to ask the database.
            
        Returns:
            str: The agent's response.
        """
        try:
            response = self.agent_executor.invoke({"input": query}, config={"callbacks": [self.langfuse_handler]})
            # The response format might vary based on the agent version, 
            # usually it returns a dict with "output" key or just the string.
            if isinstance(response, dict) and "output" in response:
                return response["output"]
            return str(response)
        except Exception as e:
            return f"Error executing query: {str(e)}"

if __name__ == "__main__":
    db_uri = r"sqlite:///data\processed\jobs.db" 
    agent = SQLAgent(db_uri)
    # print(agent.run("How many rows are in the main table?"))