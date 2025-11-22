import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

def inspect_new_wiki():
    urls = [
        "https://en.wikipedia.org/wiki/Rolling_Stone%27s_500_Greatest_Songs_of_All_Time",
        "https://en.wikipedia.org/wiki/List_of_best-selling_singles_in_the_United_States",
        "https://en.wikipedia.org/wiki/List_of_Spotify_streaming_records"
    ]
    
    for url in urls:
        print(f"\n--- Inspecting {url} ---")
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT})
            soup = BeautifulSoup(r.text, "html.parser")
            tables = soup.find_all("table", class_="wikitable")
            print(f"Found {len(tables)} wikitables.")
            
            for i, t in enumerate(tables[:3]): # Check first 3
                headers = [th.get_text(" ", strip=True) for th in t.find_all("th")]
                print(f"Table {i} headers: {headers}")
                rows = t.find_all("tr")
                if len(rows) > 1:
                    first_row = [td.get_text(" ", strip=True) for td in rows[1].find_all(["td", "th"])]
                    print(f"Table {i} row 1: {first_row}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    inspect_new_wiki()

