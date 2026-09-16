import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.search_engine import SourcingSearchEngine

st.set_page_config(page_title="Semicon Sourcing Hub", page_icon="⚡", layout="wide")


@st.cache_resource
def get_engine():
    return SourcingSearchEngine()


st.title("⚡ Semiconductor Supplier Sourcing Engine")
st.caption("Hybrid retrieval (dense semantics + BM25) with hierarchical category and geography filters")

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Failed to initialize search engine: {e}")
    st.stop()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Procurement Filters")

expo = st.sidebar.selectbox(
    "Trade show",
    ["All", "Taiwan expo", "China expo", "Japan expo", "Korea expo", "Europe expo", "India expo"],
    help="Which SEMICON show the company exhibits at - not where it is based.",
)

countries = engine.available_countries()
hq_countries = st.sidebar.multiselect(
    "Supplier HQ country",
    countries,
    help="Where the company is actually headquartered. 'Unknown' means the "
         "exhibitor profile listed no location.",
)

st.sidebar.markdown("**Category**")
l1_options = engine.level1_categories()
l1_lookup = {c["name"]: c["id"] for c in l1_options}
l1_choice = st.sidebar.selectbox("Level 1", ["All"] + list(l1_lookup.keys()))

cat_l1_ids, cat_l2_ids = None, None
if l1_choice != "All":
    cat_l1_ids = [l1_lookup[l1_choice]]
    # Level 2 options are narrowed to those actually co-occurring with this L1
    l2_names = engine.level2_names_for_l1(l1_lookup[l1_choice])
    if l2_names:
        l2_lookup = {name: cid for cid, name in l2_names}
        l2_choice = st.sidebar.multiselect("Level 2 (optional)", list(l2_lookup.keys()))
        if l2_choice:
            cat_l2_ids = [l2_lookup[n] for n in l2_choice]

max_results = st.sidebar.slider("Max results", 3, 25, 8)

active = []
if expo != "All":
    active.append(expo)
if hq_countries:
    active.append(f"HQ: {', '.join(hq_countries)}")
if l1_choice != "All":
    active.append(l1_choice)
if active:
    st.sidebar.success("Active filters: " + " • ".join(active))

# --------------------------------------------------------------- main panel
query = st.text_input(
    "Search supplier pool",
    placeholder="e.g., UHP nitrogen gas, ISO Class 3 cleanroom, wafer defect inspection, burn-in test",
)

if query:
    with st.spinner("Running hybrid search..."):
        results = engine.search(
            query=query,
            expo=expo,
            hq_countries=hq_countries or None,
            cat_l1_ids=cat_l1_ids,
            cat_l2_ids=cat_l2_ids,
            limit=max_results,
        )

    if results:
        st.subheader(f"Top {len(results)} supplier matches")
        for item in results:
            with st.container(border=True):
                # Create a 70/30 column split for desktop
                col_info, col_cat = st.columns([7, 3])

                with col_info:
                    st.markdown(f"### [{item['company_name']}]({item['url']})")

                    meta = [f"**Score** `{item['score']}`"]
                    if item.get("hq_location"):
                        meta.append(f"**HQ** {item['hq_location']}")
                    elif item.get("hq_country") and item["hq_country"] != "Unknown":
                        meta.append(f"**HQ** {item['hq_country']}")
                    if item.get("location"):
                        meta.append(f"**Show** {item['location']}")
                    st.markdown(" &nbsp;|&nbsp; ".join(meta))

                    if item.get("website"):
                        st.markdown(f"🌐 [{item['website']}]({item['website']})")

                    # "Show More" Expander for Overview
                    about_text = item.get("about")
                    if about_text and about_text != "Semiconductor technology and equipment supplier.":
                        with st.expander("📖 Show Overview"):
                            st.write(about_text)
                    else:
                        st.caption("No detailed overview provided.")

                with col_cat:
                    cat_tree = item.get("cat_tree", [])
                    if cat_tree:
                        st.markdown("##### 🏷️ Categories")
                        tree_md = ""
                        for branch in cat_tree:
                            # Print the L1 Parent
                            tree_md += f"- **{branch.get('l1_name', 'Unknown')}**\n"
                            # Print its specific L2 Children
                            for child in branch.get("children", []):
                                tree_md += f"  - {child.get('name', '')}\n"
                        st.markdown(tree_md)
                        
    else:
        st.warning(
            "No suppliers matched. Try loosening a filter - the category and HQ "
            "filters are strict (pre-filtered), so a narrow combination can return nothing."
        )
else:
    st.info("Enter a procurement query to run a hybrid search across the ingested catalog.")