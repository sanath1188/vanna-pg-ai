import vanna
import logging
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from dotenv import load_dotenv

load_dotenv()

# ================================
# CONFIGURATION
# ================================
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT"),
}

# ================================
# LOGGING SETUP
# ================================
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# ================================
# INITIALIZE VANNA AI
# ================================
log.info("Initializing Vanna AI...")

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

vn = MyVanna(config={"api_key": API_KEY, "model": MODEL_NAME})
log.info("Vanna AI initialized successfully!")

# Connect to PostgreSQL
vn.connect_to_postgres(
    host=DATABASE_CONFIG["host"],
    dbname=DATABASE_CONFIG["dbname"],
    user=DATABASE_CONFIG["user"],
    password=DATABASE_CONFIG["password"],
    port=DATABASE_CONFIG["port"],
)
log.info("Connected to PostgreSQL!")

# ================================
# ASK VANNA AI A QUESTION
# ================================
question = "Get list of users who have been active in the last month"
log.info(f"Asking Vanna AI: {question}")

try:
    generated_sql = vn.ask(question)
    log.info(f"Generated SQL:\n{generated_sql}")

    # Execute the SQL Query
    df_result = vn.run_sql(generated_sql)
    log.info("Query executed successfully! Preview:")
    log.info(df_result.head(10))

except Exception as e:
    log.error(f"Error: {e}")
