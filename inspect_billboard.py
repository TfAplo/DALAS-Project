from playwright.sync_api import sync_playwright

def inspect_billboard():
    url = "https://www.billboard.com/charts/greatest-hot-100-singles/"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print(f"Navigating to {url}...")
        page.goto(url)
        
        # Print title
        print("Page Title:", page.title())
        
        # Try to find chart items
        # Billboard usually uses classes like 'o-chart-results-list-row-container' or similar
        items = page.locator(".o-chart-results-list-row-container")
        count = items.count()
        print(f"Found {count} chart row containers.")
        
        if count > 0:
            first = items.first
            print("First item text:", first.inner_text())
            
        # Dump some HTML if no items found
        if count == 0:
            print("HTML Dump (first 1000 chars):")
            print(page.content()[:1000])
            
        browser.close()

if __name__ == "__main__":
    inspect_billboard()

