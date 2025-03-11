import vanna
import pandas as pd
import glob
import os
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
METABASE_QUERIES_FILE = os.getenv("METABASE_QUERIES_FILE")
NOTION_DOCS_FOLDER = os.getenv("NOTION_DOCS_FOLDER")
DB_SCHEMA_FILE = os.getenv("DB_SCHEMA_FILE")

# ================================
# LOGGING SETUP
# ================================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# ================================
# STEP 1: INITIALIZE VANNA AI
# ================================
log.info("Initializing Vanna AI...")

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

# Initialize Vanna with OpenAI API Key
vn = MyVanna(config={"api_key": API_KEY, "model": MODEL_NAME})

log.info("Vanna AI initialized successfully!")

# Connect to PostgreSQL
log.info("Connecting to PostgreSQL database...")
try:
    vn.connect_to_postgres(
        host=DATABASE_CONFIG["host"],
        dbname=DATABASE_CONFIG["dbname"],
        user=DATABASE_CONFIG["user"],
        password=DATABASE_CONFIG["password"],
        port=DATABASE_CONFIG["port"],
    )
    log.info("Successfully connected to PostgreSQL!")
except Exception as e:
    log.error(f"Failed to connect to PostgreSQL: {e}")

# ================================
# STEP 2: TRAIN VANNA AI WITH DATABASE SCHEMA
# ================================
def train_database_schema():
    log.info("Starting training with database schema...")
    
    if not os.path.exists(DB_SCHEMA_FILE):
        log.warning("[!] Database schema file not found. Skipping...")
        return

    df_schema = pd.read_csv(DB_SCHEMA_FILE)
    log.info(f"Loaded {len(df_schema)} schema records from {DB_SCHEMA_FILE}")

    for _, row in df_schema.iterrows():
        table = row.get("table_name", "Unknown")
        column = row.get("column_name", "Unknown")
        data_type = row.get("data_type", "Unknown")
        vn.train(ddl=f"Table: {table}, Column: {column}, Type: {data_type}")

    log.info("Finished training with database schema.")

train_database_schema()

# ================================
# STEP 3: TRAIN VANNA AI WITH METABASE QUERIES
# ================================
def train_metabase_queries():
    log.info("Starting training with Metabase queries...")

    if not os.path.exists(METABASE_QUERIES_FILE):
        log.warning("[!] Metabase queries file not found. Skipping...")
        return

    df_queries = pd.read_csv(METABASE_QUERIES_FILE)
    log.info(f"Loaded {len(df_queries)} Metabase queries.")

    for _, row in df_queries.iterrows():
        query = row.get("query", "")
        description = row.get("name", "No description")
        if query.strip():
            vn.train(sql=query, question=description)

    log.info("Finished training with Metabase queries.")

train_metabase_queries()

# ================================
# STEP 4: TRAIN VANNA AI WITH NOTION DOCUMENTATION
# ================================
def train_notion_docs():
    log.info("Starting training with Notion documentation...")

    if not os.path.exists(NOTION_DOCS_FOLDER):
        log.warning("[!] Notion docs folder not found. Skipping...")
        return

    notion_files = glob.glob(f"{NOTION_DOCS_FOLDER}/*.md")
    log.info(f"Found {len(notion_files)} Notion documentation files.")

    for file in notion_files:
        with open(file, "r", encoding="utf-8") as f:
            vn.train(documentation=f.read())

    log.info("Finished training with Notion documentation.")

train_notion_docs()

# ================================
# STEP 5: TEST VANNA AI
# ================================
log.info("Testing Vanna AI with a sample question...")
test_query = "Show me the top 10 users by purchases in the last 30 days"

try:
    generated_sql = vn.ask(test_query)
    log.info("Successfully generated SQL query:")
    log.info(f"\n{generated_sql}")
except Exception as e:
    log.error(f"Error generating SQL query: {e}")

# ================================
# STEP 6: EXECUTE THE GENERATED QUERY
# ================================
def execute_query(query):
    log.info("Executing generated SQL query...")
    try:
        df = vn.run_sql(query)
        log.info("Query executed successfully! Preview:")
        log.info(df.head(10))
    except Exception as e:
        log.error(f"Error executing query: {e}")

execute_query(generated_sql)

log.info("[✅] Training complete! Vanna AI is now ready to answer SQL queries.")
