import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)

def inspect_rolling_stone_wiki():
    url = "https://en.wikipedia.org/wiki/Rolling_Stone%27s_500_Greatest_Songs_of_All_Time"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers)
    
    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    print(f"Found {len(tables)} wikitables.")
    
    for i, tbl in enumerate(tables):
        print(f"\nTable {i}:")
        tr = tbl.find("tr")
        if tr:
            headers = [th.get_text(" ", strip=True) for th in tr.find_all(["th", "td"])]
            print("Headers:", headers)
        
        # Print first data row
        rows = tbl.find_all("tr")
        if len(rows) > 1:
            cells = [td.get_text(" ", strip=True) for td in rows[1].find_all(["td", "th"])]
            print("Row 1:", cells)

if __name__ == "__main__":
    inspect_rolling_stone_wiki()

