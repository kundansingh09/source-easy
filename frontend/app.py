import streamlit as st
import sys
import os

# Add root folder to sys.path so backend imports cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.search_engine import SourcingSearchEngine

st.set_page_config(
    page_title="Semicon Sourcing Hub",
    page_icon="⚡",
    layout="wide"
)

# Cache the search engine instance to avoid re-embedding on every rerun
@st.cache_resource
def get_engine():
    return SourcingSearchEngine()

st.title("⚡ Semiconductor Supplier Sourcing Engine")
st.caption("AI-Powered Hybrid Search (Dense Semantics + BM25 Keywords)")

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Failed to initialize search engine: {e}")
    st.stop()

# Sidebar for Filters
st.sidebar.header("Procurement Filters")
location_filter = st.sidebar.selectbox(
    "Target Geography / Sourcing Cluster",
    ["All", "Taiwan", "Taiwan expo", "India", "USA", "Japan"]
)
max_results = st.sidebar.slider("Max Results", min_value=3, max_value=20, value=5)

# Main Query Input
query = st.text_input(
    "Search Supplier Pool",
    placeholder="e.g., UHP nitrogen gas, ISO Class 3 cleanroom, packaging, wafer testing"
)

if query:
    with st.spinner("Searching vector index..."):
        results = engine.search(query=query, location_filter=location_filter, limit=max_results)

    if results:
        st.subheader(f"Top {len(results)} Supplier Matches")
        for item in results:
            with st.container():
                st.markdown(f"### [{item['company_name']}]({item['url']})")
                if item.get("hq_location"):
                    st.caption(f"🏢 **Headquarters:** {item['hq_location']}")
                st.write(f"**Relevance Score:** `{item['score']}` | **Cluster / Location:** `{item['location']}`")
                st.write(item["about"])
                st.markdown("---")
    else:
        st.warning("No suppliers found matching the criteria. Try broader search terms.")
else:
    st.info("Enter a procurement query above to run a hybrid search across the ingested catalog.")