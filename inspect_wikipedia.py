import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)

def inspect_wikipedia():
    url = "https://en.wikipedia.org/wiki/List_of_best-selling_singles"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers)
    print(f"Status: {resp.status_code}")
    
    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    print(f"Found {len(tables)} wikitables.")
    
    if len(tables) == 0:
        # Try finding any table
        all_tables = soup.find_all("table")
        print(f"Found {len(all_tables)} total tables.")
        if len(all_tables) > 0:
            print("First table classes:", all_tables[0].get("class"))

    for i, tbl in enumerate(tables):
        print(f"\nTable {i}:")
        # Print header row
        tr = tbl.find("tr")
        if tr:
            headers = [th.get_text(" ", strip=True) for th in tr.find_all(["th", "td"])]
            print("Headers:", headers)
        
        # Print first data row
        rows = tbl.find_all("tr")
        if len(rows) > 1:
            cells = [td.get_text(" ", strip=True) for td in rows[1].find_all(["td", "th"])]
            print("Row 1:", cells[:5])

if __name__ == "__main__":
    inspect_wikipedia()
