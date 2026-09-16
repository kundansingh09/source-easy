import json
import os
import threading
from qdrant_client import QdrantClient, models

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
FALLBACK_ABOUT = "Semiconductor technology and equipment supplier."


class SourcingSearchEngine:
    def __init__(self, data_path="data/semi_suppliers.json",
                 taxonomy_path="data/categories.json"):
        self.data_path = data_path
        self.taxonomy_path = taxonomy_path
        self.client = QdrantClient(":memory:")
        self.collection_name = "semicon_suppliers"
        self.taxonomy = self._load_taxonomy()
        self.lock = threading.Lock()
        self._init_collection()

    # ------------------------------------------------------------- loading

    def _load_taxonomy(self):
        if not os.path.exists(self.taxonomy_path):
            return {"level1": [], "level2": []}
        with open(self.taxonomy_path) as f:
            return json.load(f)

    def _build_text_chunk(self, item):
        """Text that gets embedded. Category NAMES are included deliberately:
        many exhibitors have no Overview at all, so their categories are the
        only semantic content they have. Without this they are unsearchable."""
        parts = [item.get("company_name", "")]

        hq = item.get("hq_location") or item.get("hq_country")
        if hq and hq != "Unknown":
            parts.append(f"HQ: {hq}")

        cats = list(item.get("cat_l1_names", [])) + list(item.get("cat_l2_names", []))
        if cats:
            parts.append("Categories: " + "; ".join(cats))

        about = item.get("about", "")
        if about and about != FALLBACK_ABOUT:
            parts.append(about)

        return " | ".join(p for p in parts if p)

    def _init_collection(self):
        if not os.path.exists(self.data_path):
            return
        with open(self.data_path) as f:
            suppliers = json.load(f)
        if not suppliers:
            return

        # Real L1 -> L2 map, built from the nested cat_tree captured at scrape
        # time (document order on the profile page), so the UI can show only
        # the subcategories that actually belong to the chosen parent.
        self.l1_to_l2 = {}
        self.l1_names_by_id = {}
        for item in suppliers:
            for node in item.get("cat_tree", []) or []:
                l1_id = node.get("l1_id")
                if l1_id is None:
                    continue
                self.l1_names_by_id.setdefault(l1_id, node.get("l1_name", str(l1_id)))
                bucket = self.l1_to_l2.setdefault(l1_id, {})
                for child in node.get("children", []) or []:
                    bucket.setdefault(child["id"], child["name"])

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
            text_chunk = self._build_text_chunk(item)
            points.append(
                models.PointStruct(
                    id=idx + 1,
                    vector={
                        "dense": models.Document(text=text_chunk, model=DENSE_MODEL),
                        "sparse": models.Document(text=text_chunk, model=SPARSE_MODEL),
                    },
                    payload={
                        "company_name": item.get("company_name", "Unknown"),
                        "location": item.get("location", "Unknown"),      # expo tag
                        "hq_location": item.get("hq_location"),           # display
                        "hq_country": item.get("hq_country", "Unknown"),  # filterable
                        "about": item.get("about", ""),
                        "website": item.get("website"),
                        "ebooth_url": item.get("ebooth_url", "#"),
                        "cat_l1_ids": item.get("cat_l1_ids", []),
                        "cat_l1_names": item.get("cat_l1_names", []),
                        "cat_l2_ids": item.get("cat_l2_ids", []),
                        "cat_l2_names": item.get("cat_l2_names", []),
                        "cat_tree": item.get("cat_tree", []),
                    },
                )
            )

        self.client.upload_points(collection_name=self.collection_name, points=points)

        # Payload indexes - negligible at ~1k points in memory, but required
        # for filters to stay fast if this ever moves to a persistent server.
        for field, schema in [
            ("location", models.PayloadSchemaType.KEYWORD),
            ("hq_country", models.PayloadSchemaType.KEYWORD),
            ("cat_l1_ids", models.PayloadSchemaType.INTEGER),
            ("cat_l2_ids", models.PayloadSchemaType.INTEGER),
        ]:
            try:
                self.client.create_payload_index(self.collection_name, field, field_schema=schema)
            except Exception:
                pass

    # ----------------------------------------------------------- filtering

    @staticmethod
    def build_filter(expo=None, hq_countries=None, cat_l1_ids=None, cat_l2_ids=None):
        """Hard metadata filter. All conditions are ANDed; values within a
        single condition are ORed (MatchAny).

        Hierarchy comes free: each point stores its own level-1 ancestors, so
        filtering on cat_l1_ids returns the whole branch including children.
        Passing cat_l2_ids narrows to specific leaves.
        """
        must = []
        if expo and expo != "All":
            must.append(models.FieldCondition(
                key="location", match=models.MatchValue(value=expo)))
        if hq_countries:
            must.append(models.FieldCondition(
                key="hq_country", match=models.MatchAny(any=list(hq_countries))))
        if cat_l2_ids:
            must.append(models.FieldCondition(
                key="cat_l2_ids", match=models.MatchAny(any=[int(i) for i in cat_l2_ids])))
        elif cat_l1_ids:
            must.append(models.FieldCondition(
                key="cat_l1_ids", match=models.MatchAny(any=[int(i) for i in cat_l1_ids])))
        return models.Filter(must=must) if must else None

    # -------------------------------------------------------------- search

    def search(self, query: str, expo=None, hq_countries=None,
               cat_l1_ids=None, cat_l2_ids=None, limit: int = 5, candidates: int = 50):
        if not query.strip():
            return []

        query_filter = self.build_filter(expo, hq_countries, cat_l1_ids, cat_l2_ids)

        # NOTE: the filter MUST go inside each prefetch branch.
        # A top-level query_filter is IGNORED when the outer query is a
        # FusionQuery - verified empirically. Putting it here gives true
        # pre-filtering: the filter constrains the vector search itself, so
        # you always get `limit` matching results rather than retrieving
        # top-k and discarding (which can return fewer than asked, or none).
        with self.lock:
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(text=query, model=DENSE_MODEL),
                        using="dense", limit=candidates, filter=query_filter,
                    ),
                    models.Prefetch(
                        query=models.Document(text=query, model=SPARSE_MODEL),
                        using="sparse", limit=candidates, filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            )

        results = []
        for r in response.points:
            p = r.payload
            results.append({
                "company_name": p.get("company_name"),
                "location": p.get("location"),
                "hq_location": p.get("hq_location"),
                "hq_country": p.get("hq_country"),
                "about": p.get("about"),
                "website": p.get("website"),
                "url": p.get("ebooth_url"),
                "categories_l1": p.get("cat_l1_names", []),
                "categories_l2": p.get("cat_l2_names", []),
                "cat_tree": p.get("cat_tree", []),
                "score": round(r.score, 4),
            })
        return results

    # ----------------------------------------------------------- UI helpers

    def level1_categories(self):
        """Prefer the scraped master taxonomy; fall back to what the ingested
        records actually contain so the UI still works without categories.json."""
        from_file = self.taxonomy.get("level1", [])
        if from_file:
            observed = set(getattr(self, "l1_names_by_id", {}).keys())
            if observed:
                return [c for c in from_file if c["id"] in observed] or from_file
            return from_file
        return [{"id": k, "name": v} for k, v in sorted(
            getattr(self, "l1_names_by_id", {}).items(), key=lambda kv: kv[1])]

    def level2_categories(self):
        return self.taxonomy.get("level2", [])

    def level2_names_for_l1(self, l1_id):
        """[(id, name), ...] of subcategories belonging to this parent."""
        bucket = getattr(self, "l1_to_l2", {}).get(int(l1_id), {})
        return sorted(bucket.items(), key=lambda kv: kv[1])

    def available_countries(self):
        """Distinct hq_country values actually present, for the filter widget."""
        seen = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection_name, limit=500, offset=offset, with_payload=["hq_country"]
            )
            for pt in points:
                c = (pt.payload or {}).get("hq_country")
                if c:
                    seen.add(c)
            if offset is None:
                break
        return sorted(seen)


if __name__ == "__main__":
    engine = SourcingSearchEngine()
    for row in engine.search("wafer defect inspection", limit=5):
        print(row["company_name"], "|", row["hq_country"], "|", row["score"])