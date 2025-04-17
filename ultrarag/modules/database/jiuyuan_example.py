import asyncio

import numpy as np
from jiuyuan_db.vector.sdk import Record

from ultrarag.modules.embedding import EmbClient
from ultrarag.modules.database.jiuyuan import JiuyuanVectorStore

async def main():
    encoder=EmbClient(url_or_path="embedding_url")
    vector_store = JiuyuanVectorStore(
        host="localhost",
        port=5432,
        user="postgres",
        password="mysecretpassword",
        db_name="postgres",
        encoder=encoder
    )

    # Define collection details.
    collection_name = "demo_collection_1"
    dimension = 3

    # Create (or recreate) the table for this collection.
    await vector_store.create(collection_name=collection_name, dimension=dimension)
    print(f"Table for collection '{collection_name}' created.")

    # Prepare some dummy records.
    dummy_payloads = [
        {
            "text": "This is a test record.",
            "embedding": np.random.rand(dimension).tolist(),  # Convert np array to list
            "meta": {"category": "test", "id": 1}
        },
        {
            "text": "Another record with different text.",
            "embedding": np.random.rand(dimension).tolist(),
            "meta": {"category": "test", "id": 2}
        },
    ]

    # Insert records into the collection.
    print("Inserting records...")
    records = [
        Record.from_text(text=node["text"], embedding=node["embedding"], meta=node["meta"])
        for node in dummy_payloads
    ]

    # Insert the records into the collection.
    await vector_store.insert_records(collection_name, records)
    print(f"Inserted {len(records)} records into collection '{collection_name}'.")

    # Perform a search using a target embedding.
    search_embedding = [1.0, 1.0, 1.0]
    results = await vector_store.search_by_embedding(collection_name, search_embedding, topn=3)
    if not results:
        print("No results found.")
    else:
        print("Search Results:")
        for result in results:
            print(f"- {result.content} | Score: {result.score:.4f}")

    await vector_store.remove(collection_name)
    print("\nCollection removed.")

    # collection_name = "demo_collection_2"
    # await vector_store.create(collection_name=collection_name, dimension=1024)
    #
    # sample_data = [
    #     {"id": 1, "text": "Apple is a fruit"},
    #     {"id": 2, "text": "The sky is blue"},
    #     {"id": 3, "text": "I love pizza"},
    # ]
    #
    # def extract_text(item):
    #     return item["text"]
    #
    # def report_progress(pct):
    #     print(f"Insertion progress: {pct:.2f}%")
    #
    # await vector_store.insert(
    #     collection=collection_name,
    #     payloads=sample_data,
    #     func=extract_text,
    #     callback=report_progress,
    #     batch_size=1
    # )
    #
    # print("\nSearch Results:")
    # results = await vector_store.search(
    #     collection=collection_name,
    #     query="What color is the sky?",
    #     topn=3
    # )
    # for result in results:
    #     print(f"- {result.content} | Score: {result.score:.4f}")
    #
    # await vector_store.remove(collection_name)
    # print("\nCollection removed.")

# Run the example
if __name__ == "__main__":
    asyncio.run(main())