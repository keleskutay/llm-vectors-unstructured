import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings

COURSES_PATH = "data/asciidoc"

# Load lesson documents
loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

# Create a text splitter
text_splitter = CharacterTextSplitter(separator="\n\n",chunk_size=1500,chunk_overlap=200)

# Split documents into chunks
chunks = text_splitter.split_documents(docs)

# Create a Neo4j vector store
neo4j_db = Neo4jVector.from_documents(
                       url=os.getenv("NEO4J_URI"),
                       username=os.getenv("NEO4J_USERNAME"),
                       password=os.getenv("NEO4J_PASSWORD"),
                       database=os.getenv("NEO4J_DATABASE"),
                       embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
                       documents=chunks,
                       index_name="chunkVector",
                       node_label="Chunk",
                       text_node_property="text",
                       embedding_node_property="embedding")