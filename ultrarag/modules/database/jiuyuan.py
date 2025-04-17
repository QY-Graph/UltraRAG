import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from jiuyuan_db.vector.sdk import Record
from jiuyuan_db.vector.sdk.client import JiuyuanVector

from ultrarag.modules.database import BaseIndex, BaseNode
from ultrarag.modules.embedding import BaseEmbedding

from loguru import logger

class JiuyuanVectorStore(BaseIndex):
    def __init__(
            self,
            host: str,
            port: int,
            user: str,
            password: str,
            db_name: str,
            encoder: BaseEmbedding,
    ) -> None:
        """
        Initialize the vector store by creating a JiuyuanVector client.

        Args:
            host (str): Database host.
            port (int): Database port.
            user (str): Database user.
            password (str): Database password.
            db_name (str): Database name.
            encoder (BaseEmbedding): Encoder used to generate document embeddings.
        """
        super().__init__()
        self.encoder = encoder
        self.client = JiuyuanVector.from_config(
            host=host,
            port=port,
            user=user,
            password=password,
            db_name=db_name,
        )

    async def create(
            self, collection_name: str, dimension: int = 1024, index_type: str = "dense", **kwargs
    ) -> None:
        """
        Create a new vector collection with the specified dimension.

        Args:
            collection_name (str): Name of the collection.
            dimension (int): Embedding dimension for the collection.
            index_type (str): Type of vector indexing ('dense' or 'hybrid'). (Currently unused.)
            **kwargs: Additional keyword arguments.
        """
        await self.client.create_table(collection_name, dimension)

    async def insert(
        self,
        collection: str,
        payloads: List[Dict[str, Any]],
        func: Callable = lambda x: x,
        method: str = "dense",
        callback: Optional[Callable[[float], None]] = None,
        batch_size: int = 10,
    ) -> None:
        """
        Insert data into the vector database while reporting progress via a callback.

        Args:
            collection (str): Collection name.
            payloads (List[Dict[str, Any]]): List of dictionaries containing data to insert.
            func (Callable): Function to extract text content from each payload.
            method (str): Vector indexing method, e.g., 'dense' or 'hybrid'.
            callback (Optional[Callable]): Callback function for reporting insertion progress.
            batch_size (int): Number of records to insert per batch.
        """
        contents = [func(item) for item in payloads]
        # [logger.info(f"content: {content}") for content in contents]
        embeddings = await self.encoder.document_encode(contents)
        # [logger.info(f"embed: {embed.get('dense_embed', None)}") for embed in embeddings]
        if len(embeddings) != len(payloads):
            raise ValueError("Embedding count does not match payload count.")
        records = []
        for content, payload, embed in zip(contents, payloads, embeddings):
            if method == "dense":
                records.append(
                    Record.from_text(text=content,
                                     embedding=embed.get("dense_embed", None),
                                     meta=payload)
                )
            elif method == "hybrid":
                records.append(
                    Record.from_text(text=content,
                                     embedding=embed.get("hybrid_embed", None),
                                     meta=payload)
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

        await self.insert_records(collection, records, callback, batch_size)

    async def insert_records(
        self,
        collection: str,
        records: List[Record],
        callback: Optional[Callable[[float], None]] = None,
        batch_size: int = 10,
    ) -> None:
        """
        Insert pre-encoded records into the collection.

        Args:
            collection (str): Collection name.
            records (List[Record]): List of Record objects.
            callback (Optional[Callable]): Progress callback.
            batch_size (int): Batch size.
        """
        total = len(records)
        for i in range(0, total, batch_size):
            batch = records[i: i + batch_size]
            await self.client.insert(collection, batch)
            if callback:
                callback((i + len(batch)) / total * 100.0)

    async def search(
        self,
        collection: Union[str, List[str]],
        query: str,
        topn: int = 5,
        method: str = "dense",
        **kwargs,
    ) -> List['BaseNode']:
        """
        Search the vector database for similar records.

        Args:
            collection (Union[str, List[str]]): Collection name or list of names.
            query (str): Input query.
            topn (int): Top N results.
            method (str): Vector indexing method.
            **kwargs: Additional options.

        Returns:
            List[BaseNode]: List of result nodes.
        """
        embedding = await self.encoder.query_encode(query)
        if method == "dense":
            query_embedding = embedding.get("dense_embed", None)
        elif method == "hybrid":
            query_embedding = embedding.get("hybrid_embed", None)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return await self.search_by_embedding(collection, query_embedding, topn)

    async def search_by_embedding(
        self,
        collection: Union[str, List[str]],
        query_embedding: List[float],
        topn: int,
    ) -> List['BaseNode']:
        """
        Search using a precomputed embedding.

        Args:
            collection (Union[str, List[str]]): Collection(s) to search.
            query_embedding (List[float]): Query embedding vector.
            topn (int): Number of top results to return.

        Returns:
            List[BaseNode]: Result nodes.
        """
        if isinstance(collection, list):
            results_list = await asyncio.gather(
                *(self.client.search(coll, query_embedding, top_k=topn) for coll in collection)
            )
            all_results = [item for sublist in results_list for item in sublist]
            all_results.sort(key=lambda x: x[1])
            results = all_results[:topn]
        else:
            results = await self.client.search(collection, query_embedding, top_k=topn)

        return [
            BaseNode(content=record.text, score=distance, payload=record.to_dict())
            for record, distance in results
        ]

    async def remove(self, collection: Union[str, List[str]]) -> None:
        """
        Remove (drop) the specified collection from the database.

        Args:
            collection (Union[str, List[str]]): Name of the collection to remove.
        """
        await self.client.drop_table(collection)

