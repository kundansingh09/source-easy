import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def explore_hierarchy():
    print("--- ADVANCED SEARCH CATEGORY TREE ---")
    search_url = "https://expo.semi.org/taiwan2026/public/AdvancedSearch.aspx"
    resp = requests.get(search_url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # Try to find the category container block
    cat_block = soup.find(id="Categories") or soup.find(class_="category-list")
    if not cat_block:
        # Fallback to finding list items that look like categories
        cat_block = soup.find('ul', class_='category-tree')
        
    print(cat_block.prettify()[:1500] if cat_block else "Could not isolate Master Category block. Please provide a snippet of the raw HTML.")

    print("\n--- EBOOTH PROFILE CATEGORIES ---")
    # AblePrint's profile which clearly has categories
    profile_url = "https://expo.semi.org/taiwan2026/public/eBooth.aspx?BoothID=653254"
    resp2 = requests.get(profile_url, headers=HEADERS)
    soup2 = BeautifulSoup(resp2.text, 'html.parser')
    
    # Find the header that says "Categories" and grab its parent/sibling
    cat_header = soup2.find(lambda tag: tag.name in ["h2", "h3", "h4", "div"] and "Categories" in tag.text)
    if cat_header and cat_header.parent:
        print(cat_header.parent.prettify()[:1500])
    else:
        print("Could not isolate Profile Category block.")

if __name__ == "__main__":
    explore_hierarchy()