import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# --- Load environment variables ---
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GRAPHRAG_LLM_API_KEY")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
INDEX_NAME = "text_units"

# Setup driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Setup embedder
embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")


# --- Save entities and relationships ---
import numpy as np


def save_to_neo4j():
    print("\n--- Loading Graph into Neo4j ---")

    # Load parquet files
    entities_path = os.path.join(OUTPUT_DIR, "entities.parquet")
    rels_path = os.path.join(OUTPUT_DIR, "relationships.parquet")

    if not os.path.exists(entities_path) or not os.path.exists(rels_path):
        print("Error: Graph files not found in /output. Run the indexing pipeline first.")
        return

    entities_df = pd.read_parquet(entities_path).dropna(subset=["title"])
    rels_df = pd.read_parquet(rels_path).dropna(subset=["source", "target"])

    # --- Normalize titles for matching ---
    def normalize_text(s):
        if pd.isna(s):
            return None
        return str(s).strip().upper()  # remove spaces, uppercase

    entities_df['title'] = entities_df['title'].apply(normalize_text)
    rels_df['source'] = rels_df['source'].apply(normalize_text)
    rels_df['target'] = rels_df['target'].apply(normalize_text)

    # --- Convert array fields to list for Neo4j ---
    for col in ['text_unit_ids']:
        if col in rels_df.columns:
            rels_df[col] = rels_df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    entities_records = entities_df.astype(object).where(pd.notna(entities_df), None).to_dict("records")
    rels_records = rels_df.astype(object).where(pd.notna(rels_df), None).to_dict("records")

    with driver.session() as session:
        # Create unique constraint on title
        session.run("CREATE CONSTRAINT entity_title IF NOT EXISTS FOR (e:Entity) REQUIRE e.title IS UNIQUE")

        # Ingest entities
        print(f"Ingesting {len(entities_records)} entities...")
        session.run(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {title: row.title})
            SET e += apoc.map.clean(row, ['title'], [])
            """,
            rows=entities_records
        )

        # Ingest relationships using title
        print(f"Ingesting {len(rels_records)} relationships...")
        failed_rels = []
        for row in rels_records:
            try:
                result = session.run(
                    """
                    MATCH (s:Entity {title: $source})
                    MATCH (t:Entity {title: $target})
                    MERGE (s)-[r:RELATIONSHIP {id: $id}]->(t)
                    SET r += apoc.map.clean($props, ['id', 'source', 'target'], [])
                    RETURN r
                    """,
                    {"source": row["source"], "target": row["target"], "id": row["id"], "props": row}
                )
                if result.single() is None:
                    failed_rels.append(row)
            except Exception as e:
                failed_rels.append({"row": row, "error": str(e)})

        if failed_rels:
            print(f"Warning: {len(failed_rels)} relationships failed to insert:")
            for f in failed_rels[:5]:
                print(f)
        else:
            print("--- All relationships ingested successfully ---")

# --- Save text units ---
def save_text_units(parquet_file="output/text_units.parquet"):
    df = pd.read_parquet(parquet_file)
    with driver.session() as session:
        for _, row in df.iterrows():
            embedding = embedder.embed_query(row["text"]) if embedder else []
            session.run(
                """
                MERGE (t:TextUnit {id: $id})
                SET t.text = $text, t.embedding = $embedding
                """,
                {"id": row["id"], "text": row["text"], "embedding": embedding}
            )
    print(f"--- Ingested {len(df)} text units into Neo4j ---")


# --- Create vector index ---
def create_vector_index(index_name="text_units"):
    with driver.session() as session:
        session.run(f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (t:TextUnit) ON (t.embedding)
        """)
    print(f"--- Vector index '{index_name}' created (or already exists) ---")



def run_query_debug(query="Whats the age of elizabeth called"):
    retriever = VectorRetriever(driver, INDEX_NAME, embedder)
    llm = OpenAILLM(model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)
    rag = GraphRAG(retriever=retriever, llm=llm)

    # Perform RAG search
    response = rag.search(query_text=query, retriever_config={"top_k": 5})

    # LLM answer
    print("\n--- LLM-Generated Answer ---")
    print(response.answer)

    # Print the context nodes used by the LLM
    if hasattr(response, "source_nodes") and response.source_nodes:
        print("\n--- Context Passed to LLM ---")
        for i, node in enumerate(response.source_nodes, 1):
            # each node is a dictionary containing at least 'text'
            text = node.get("text") if isinstance(node, dict) else str(node)
            print(f"{i}. {text}\n")



# --- Main Execution ---
if __name__ == "__main__":
    # save_to_neo4j()
    # save_text_units("output/text_units.parquet")
    # create_vector_index(INDEX_NAME)
    run_query_debug()
    driver.close()
