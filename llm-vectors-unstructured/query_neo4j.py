import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
#from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph

llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = llm.embeddings.create(
        input="What does Hallucination mean?",
        model="text-embedding-ada-002"
    )

embedding = response.data[0].embedding

# Connect to Neo4j
#graph = GraphDatabase.driver(uri=os.getenv("NEO4J_URI"),
#                            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
#                           database=os.getenv("NEO4J_DATABASE")
#
#                              ) 
graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), 
                   username=os.getenv("NEO4J_USERNAME"), 
                   password=os.getenv("NEO4J_PASSWORD"), 
                   database=os.getenv("NEO4J_DATABASE"))

# Run query
result = graph.query(f"CALL db.index.vector.queryNodes('chunkVector', 6, {embedding}) YIELD node, score RETURN node.text, score")

# Display results
for row in result:
    print(row)
   
