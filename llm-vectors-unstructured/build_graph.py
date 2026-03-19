import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from openai import OpenAI
from neo4j import GraphDatabase

COURSES_PATH = "data/asciidoc"

loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(docs)

# Create a function to get the embedding
def get_embedding(llm: OpenAI, text: str):
    response = llm.embeddings.create(input=text,model="text-embedding-ada-002")
    return response.data[0].embedding

# Create a function to get the course data
def get_course_data(llm, chunk):
    chunk_split = chunk.metadata["source"].split(os.path.sep)
    chunk_page_content = chunk.page_content

    data = dict()

    data["course"] = chunk_split[3]
    data["module"] = chunk_split[5]
    data["lesson"] = chunk_split[7]
    data["url"] = f"https://graphacademy.neo4j.com/courses/{data["course"]}/{data['module']}/{data["lesson"]}/"
    data["text"] = chunk_page_content
    data["embedding"] = get_embedding(llm, data["text"])

    return data

# Create OpenAI object
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#test = get_course_data(openai, chunks[0])
#print(test["course"])

# Connect to Neo4j
driver = GraphDatabase.driver(
    uri=os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    database=os.getenv("NEO4J_DATABASE")
)

# Create a function to run the Cypher query
def create_data_model(tx, data):
    tx.run(""" MERGE(c:Course {name: $course})
               MERGE(c)-[:HAS_MODULE]->(m:Module {name: $module})
               MERGE(m)-[:HAS_LESSON]->(l:Lesson {name: $lesson, url: $url})
               MERGE(l)-[:CONTAINS]->(p:Paragraph {text: $text})
               WITH p
               CALL db.create.setNodeVectorProperty(p, "embedding", $embedding)
           """, data)

# Iterate through the chunks and create the graph
for chunk in chunks:
    with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
        session.execute_write(
            create_data_model,
            get_course_data(openai, chunk)   
        )

# Close the neo4j driver
driver.close()