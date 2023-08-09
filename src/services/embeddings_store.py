
from typing import TypedDict, cast, Optional
from pymilvus import (
    SearchResult,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from config import settings

ID_FIELD = "id"
RESOURCE_NAME_FIELD = "resource_name"
RESOURCE_ID_FIELD = "resource_id"
DATA_FIELD = "data"
EMBEDDINGS_FIELD = "embeddings"
PAYLOAD_FIELD = "payload"


class ResourceChunkInfo(TypedDict):
    id: Optional[int]
    resource_name: str
    resource_id: str
    data: str
    embeddings: list[float]
    payload: str


class CollectionEmbeddingsStore:
    def __init__(self, collection_name: str, host: str = settings.milvus.host, port: str = settings.milvus.port):
        self.collection_name = self.make_guid_compatible(collection_name)
        self.host = host
        self.port = port
        self.connection_alias = "default"
        self.collection: Collection

    def make_guid_compatible(self, collection_name: str) -> str:
        return "_" + collection_name.replace("-", "_")

    def setup(self, create_index: bool = False):
        fields = [
            FieldSchema(name=ID_FIELD, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=RESOURCE_NAME_FIELD, dtype=DataType.VARCHAR, max_length=settings.database.resource_name_size),
            FieldSchema(name=RESOURCE_ID_FIELD, dtype=DataType.VARCHAR, max_length=settings.database.resource_id_size),
            FieldSchema(name=DATA_FIELD, dtype=DataType.VARCHAR, max_length=settings.database.data_size),
            FieldSchema(name=EMBEDDINGS_FIELD, dtype=DataType.FLOAT_VECTOR, dim=settings.database.embedding_size),
            FieldSchema(name=PAYLOAD_FIELD, dtype=DataType.VARCHAR, max_length=settings.database.payload_size)
        ]

        resource_chunk_schema = CollectionSchema(fields, "Schema for holding resource chunk embeddings")

        self.collection = Collection(self.collection_name, resource_chunk_schema, consistency_level="Bounded")

        if create_index:
            try:
                index = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "IP",
                    "params": {"nlist": 128},
                }

                self.collection.create_index(EMBEDDINGS_FIELD, index)
            except:
                pass
            finally:
                pass

    def drop_collection(self):
        utility.drop_collection(self.collection_name)

    def delete_resource_chunks(self, resource_id: str):
        self.collection.load()  # type: ignore

        result = self.collection.query(
            expr=f"{RESOURCE_ID_FIELD} == \"{resource_id}\"",
            output_fields=[ID_FIELD, RESOURCE_ID_FIELD]
        )

        ids_to_delete = [r[ID_FIELD] for r in result]

        self.collection.delete(f"{ID_FIELD} in [{','.join([str(id) for id in ids_to_delete])}]")  # type: ignore

    def insert_resource_chunks(self, entities: list[ResourceChunkInfo]):
        if self.collection is None:
            raise ValueError("Collection not created.")

        formatted_entities = [
            [e[RESOURCE_NAME_FIELD] for e in entities],
            [e[RESOURCE_ID_FIELD] for e in entities],
            [e[DATA_FIELD] for e in entities],
            [e[EMBEDDINGS_FIELD] for e in entities],
            [e[PAYLOAD_FIELD] for e in entities],
        ]

        self.collection.insert(formatted_entities)

        self.collection.flush()

    def search_similar_chunks(self, query_vectors: list[float], limit: int = 5) -> list[tuple[ResourceChunkInfo, float]]:
        self.collection.load()

        if self.collection is None:
            raise ValueError(
                "Collection not created. Please call create_collection() method first.")

        result = self.collection.search(
            [query_vectors],
            "embeddings",
            {"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=[ID_FIELD, DATA_FIELD, RESOURCE_ID_FIELD, RESOURCE_NAME_FIELD, PAYLOAD_FIELD],
            consistency_level="Bounded"
        )

        result = cast(SearchResult, result)

        return [(cast(ResourceChunkInfo, {
            ID_FIELD: r.entity.id,
            RESOURCE_NAME_FIELD: r.entity.resource_name,
            RESOURCE_ID_FIELD: r.entity.resource_id,
            DATA_FIELD: r.entity.data,
            PAYLOAD_FIELD: r.entity.payload
        }), cast(float, r.distance)) for r in result[0]]

    def get_chunks_data(self, chunk_ids: list[str]) -> list[ResourceChunkInfo]:
        self.collection.load()

        if self.collection is None:
            raise ValueError(
                "Collection not created. Please call create_collection() method first.")

        result = self.collection.query(
            expr=f"{ID_FIELD} in [{','.join([str(id) for id in chunk_ids])}]",
            output_fields=[ID_FIELD, DATA_FIELD, RESOURCE_ID_FIELD, RESOURCE_NAME_FIELD, PAYLOAD_FIELD]
        )

        return [cast(ResourceChunkInfo, {
            ID_FIELD: r[ID_FIELD],
            RESOURCE_NAME_FIELD: r[RESOURCE_NAME_FIELD],
            RESOURCE_ID_FIELD: r[RESOURCE_ID_FIELD],
            DATA_FIELD: r[DATA_FIELD],
            PAYLOAD_FIELD: r[PAYLOAD_FIELD]
        }) for r in result]
