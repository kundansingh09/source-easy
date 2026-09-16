import json
import os
from qdrant_client import QdrantClient, models

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"

class SourcingSearchEngine:
    def __init__(self, data_path="data/semi_suppliers.json"):
        self.data_path = data_path
        # Local in-memory Qdrant instance - zero-dependency local dev / demo
        self.client = QdrantClient(":memory:")
        self.collection_name = "semicon_suppliers"
        self._init_collection()

    def _init_collection(self):
        if not os.path.exists(self.data_path):
            return

        with open(self.data_path, "r") as f:
            suppliers = json.load(f)

        if not suppliers:
            return

        # Named dense + sparse vectors on the same collection = true hybrid search
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.client.get_embedding_size(DENSE_MODEL),
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

        points = []
        for idx, item in enumerate(suppliers):
            company_name = item.get("company_name", "Unknown")
            location = item.get("location", "Taiwan")  # expo/sourcing-cluster tag
            hq_location = item.get("hq_location")  # real company HQ, may be missing
            about = item.get("about", "")
            hq_part = f" (HQ: {hq_location})" if hq_location else ""
            text_chunk = f"{company_name} - {location}{hq_part}: {about}"

            points.append(
                models.PointStruct(
                    id=idx + 1,
                    vector={
                        # embedding happens automatically on upload - no manual
                        # embedding calls needed
                        "dense": models.Document(text=text_chunk, model=DENSE_MODEL),
                        "sparse": models.Document(text=text_chunk, model=SPARSE_MODEL),
                    },
                    payload={
                        "company_name": company_name,
                        "location": location,
                        "hq_location": hq_location,
                        "about": about,
                        "ebooth_url": item.get("ebooth_url", "#"),
                    },
                )
            )

        self.client.upload_points(collection_name=self.collection_name, points=points)

    def search(self, query: str, location_filter: str = "All", limit: int = 5):
        if not query.strip():
            return []

        query_filter = None
        if location_filter and location_filter != "All":
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="location", match=models.MatchValue(value=location_filter)
                    )
                ]
            )

        # Hybrid retrieval: fetch top candidates from each vector, fuse with RRF
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query, model=DENSE_MODEL),
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=SPARSE_MODEL),
                    using="sparse",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
        )

        formatted_results = []
        for r in response.points:
            formatted_results.append(
                {
                    "company_name": r.payload.get("company_name"),
                    "location": r.payload.get("location"),
                    "hq_location": r.payload.get("hq_location"),
                    "about": r.payload.get("about"),
                    "url": r.payload.get("ebooth_url"),
                    "score": round(r.score, 4),
                }
            )

        return formatted_results


if __name__ == "__main__":
    # Quick smoke test: python backend/search_engine.py
    engine = SourcingSearchEngine()
    for row in engine.search("UHP nitrogen gas supplier"):
        print(row)