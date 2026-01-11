from langchain_openai import ChatOpenAI
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
)
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0)