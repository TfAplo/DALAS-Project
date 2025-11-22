"""
Scrape official chart websites listed in data/chart_official_sites.csv.
...
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
import argparse
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)


@dataclass
class ChartRow:
    source_domain: str
    chart: str
    region: Optional[str]
    date: Optional[str]        # YYYY-MM-DD if available
    position: int
    title: str
    artist: str
    url: Optional[str]         # track URL if available
    scraped_at: str            # ISO timestamp


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


# ---------- Generic helpers ----------
def requests_get(url: str, timeout: int = 30) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


def parse_table_to_rows(
    table_soup: BeautifulSoup,
    source_domain: str,
    chart_name: str,
    region: Optional[str],
    date: Optional[str],
    scraped_at: str,
) -> List[ChartRow]:
    rows_out: List[ChartRow] = []
    header_cells = []
    header_tr = table_soup.find("tr")
    if header_tr:
        ths = header_tr.find_all(["th", "td"])
        header_cells = [th.get_text(" ", strip=True).lower() for th in ths]

    idx_pos = None
    idx_title = None
    idx_artist = None
    candidates_title = {"title", "song", "track", "single", "album"}
    candidates_artist = {"artist", "performer", "by", "singer"}
    candidates_pos = {"pos", "position", "rank", "#"}

    if header_cells:
        for i, h in enumerate(header_cells):
            hnorm = h.replace("no.", "#").strip()
            if idx_pos is None and any(tok in hnorm for tok in candidates_pos):
                idx_pos = i
            if idx_title is None and any(tok in hnorm for tok in candidates_title):
                idx_title = i
            if idx_artist is None and any(tok in hnorm for tok in candidates_artist):
                idx_artist = i

    trs = table_soup.find_all("tr")
    start_idx = 1 if header_cells else 0
    for idx_row, tr in enumerate(trs[start_idx:], start=1):
        tds = tr.find_all(["td", "th"])
        if not tds or len(tds) < 2:
            continue
        pos_text = None
        title_text = None
        artist_text = None

        if idx_pos is not None and idx_pos < len(tds):
            pos_text = tds[idx_pos].get_text(" ", strip=True)
        if idx_title is not None and idx_title < len(tds):
            title_text = tds[idx_title].get_text(" ", strip=True)
        if idx_artist is not None and idx_artist < len(tds):
            artist_text = tds[idx_artist].get_text(" ", strip=True)

        if idx_pos is None:
            # Fallback: Try to parse rank from first cell, OR use row index
            first_cell = tds[0].get_text(" ", strip=True)
            if first_cell.isdigit():
                pos_text = first_cell
            else:
                # Use row index as position if no explicit rank column found
                position = idx_row
        else:
             try:
                 position = int("".join(ch for ch in (pos_text or "") if ch.isdigit()))
             except:
                 # Fallback to row index if parsing fails
                 position = idx_row

        if idx_pos is None and 'position' not in locals():
             position = idx_row

        if idx_title is None:
            if len(tds) >= 2:
                guess = tds[1].get_text(" ", strip=True)
                if " - " in guess:
                    title_guess, _, artist_guess = guess.partition(" - ")
                    title_text = title_text or title_guess.strip()
                    artist_text = artist_text or artist_guess.strip()
                elif " – " in guess:
                    title_guess, _, artist_guess = guess.partition(" – ")
                    title_text = title_text or title_guess.strip()
                    artist_text = artist_text or artist_guess.strip()
                else:
                    title_text = title_text or guess

        rows_out.append(
            ChartRow(
                source_domain=source_domain,
                chart=chart_name,
                region=region,
                date=date,
                position=position,
                title=title_text or "",
                artist=artist_text or "",
                url=None,
                scraped_at=scraped_at,
            )
        )
    return rows_out


def scrape_turntablecharts(url: str, chart_name: str) -> List[ChartRow]:
    domain = get_domain(url)
    ts = utcnow_iso()
    rows: List[ChartRow] = []
    try:
        resp = requests_get(url, timeout=45)
    except Exception:
        return rows

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table")
    for tbl in tables:
        rows.extend(parse_table_to_rows(tbl, domain, chart_name, region=None, date=None, scraped_at=ts))
        if rows:
            rows = dedup_by_position(rows)
            return rows[:200]

    items = soup.select("li, div")
    scratch: List[Tuple[int, str, str]] = []
    for el in items:
        txt = el.get_text(" ", strip=True)
        parts = txt.split()
        if not parts:
            continue
        try:
            rank = int(parts[0].rstrip(".#"))
        except Exception:
            continue
        if " - " in txt:
            title, _, artist = txt.partition(" - ")
        elif " – " in txt:
            title, _, artist = txt.partition(" – ")
        else:
            continue
        scratch.append((rank, title.strip(), artist.strip()))
    if scratch:
        for pos, title, artist in scratch:
            rows.append(
                ChartRow(
                    source_domain=domain,
                    chart=chart_name,
                    region=None,
                    date=None,
                    position=pos,
                    title=title,
                    artist=artist,
                    url=None,
                    scraped_at=ts,
                )
            )
    rows = dedup_by_position(rows)
    return rows[:200]


def scrape_billboard_greatest(url: str, chart_name: str) -> List[ChartRow]:
    if "charts/" not in url:
        return []

    from playwright.sync_api import sync_playwright
    domain = get_domain(url)
    ts = utcnow_iso()
    rows: List[ChartRow] = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        # Attempt without specific UA in context to mimic default behavior which worked in inspection
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            
            try:
                page.wait_for_selector(".o-chart-results-list-row-container", timeout=20000)
            except Exception:
                pass 

            extracted = page.eval_on_selector_all(
                ".o-chart-results-list-row-container",
                """containers => containers.map(c => {
                    const rankEl = c.querySelector("span.c-label"); 
                    let rank = rankEl ? rankEl.innerText.trim() : "";
                    
                    // Fallback for rank
                    if (!rank || isNaN(parseInt(rank))) {
                         // Try finding simple text node
                         const txt = c.innerText;
                         const match = txt.match(/^\s*(\d+)/);
                         if (match) rank = match[1];
                    }

                    // Title is usually in h3#title-of-a-story
                    const titleEl = c.querySelector("h3#title-of-a-story");
                    let title = titleEl ? titleEl.innerText.trim() : "";

                    // Artist is typically in a span following the title
                    let artist = "";
                    if (titleEl) {
                        // Check sibling
                        const sibling = titleEl.nextElementSibling;
                        if (sibling && sibling.tagName === 'SPAN') {
                            artist = sibling.innerText.trim();
                        }
                    }
                    
                    return { rank, title, artist };
                })"""
            )
            
            for item in extracted:
                try:
                    rank = int(item["rank"])
                    title = item["title"]
                    artist = item["artist"]
                    if title and artist:
                        rows.append(
                            ChartRow(
                                source_domain=domain,
                                chart=chart_name,
                                region="US",
                                date=None,
                                position=rank,
                                title=title,
                                artist=artist,
                                url=None,
                                scraped_at=ts,
                            )
                        )
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"[billboard] Error scraping {url}: {e}", file=sys.stderr)
        finally:
            browser.close()
            
    return rows


def scrape_wikipedia_list(url: str, chart_name: str) -> List[ChartRow]:
    domain = get_domain(url)
    ts = utcnow_iso()
    rows: List[ChartRow] = []
    try:
        resp = requests_get(url)
    except Exception:
        return rows
        
    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    
    for tbl in tables:
        new_rows = parse_table_to_rows(tbl, domain, chart_name, region=None, date=None, scraped_at=ts)
        if new_rows:
             rows.extend(new_rows)
             
    return dedup_by_position(rows)


def dedup_by_position(rows: List[ChartRow]) -> List[ChartRow]:
    seen = set()
    out: List[ChartRow] = []
    for r in rows:
        key = (r.position, r.title.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def browser_extract_first_good_table(url: str) -> List[List[str]]:
    from playwright.sync_api import sync_playwright

    def table_rows_to_lists(page, table_selector: str) -> List[List[str]]:
        return page.eval_on_selector_all(
            table_selector + " tr",
            """els => els.map(tr => Array.from(tr.querySelectorAll('th,td')).map(td => td.innerText.trim()))""",
        )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(user_agent=USER_AGENT, accept_downloads=True, locale="en-US")
        page = context.new_page()
        page.set_default_timeout(30000)
        page.goto(url, wait_until="domcontentloaded")

        tables = page.locator("table")
        count = tables.count()
        for i in range(count):
            selector = f"table:nth-of-type({i+1})"
            try:
                rows = table_rows_to_lists(page, selector)
            except Exception:
                continue
            if not rows or len(rows) < 11:
                continue
            if max(len(r) for r in rows) < 2:
                continue
            browser.close()
            return rows

        browser.close()
    return []


def generic_browser_chart(url: str, chart_name: str, region: Optional[str] = None) -> List[ChartRow]:
    domain = get_domain(url)
    ts = utcnow_iso()
    rows_text = browser_extract_first_good_table(url)
    if not rows_text:
        return []

    header = rows_text[0]
    body = rows_text[1:] if len(rows_text) > 1 else []
    header_lower = [h.lower() for h in header]

    idx_pos = None
    idx_title = None
    idx_artist = None
    candidates_title = {"title", "song", "track", "single", "album"}
    candidates_artist = {"artist", "performer", "by", "singer"}
    candidates_pos = {"pos", "position", "rank", "#"}

    for i, h in enumerate(header_lower):
        hnorm = h.replace("no.", "#").strip()
        if idx_pos is None and any(tok in hnorm for tok in candidates_pos):
            idx_pos = i
        if idx_title is None and any(tok in hnorm for tok in candidates_title):
            idx_title = i
        if idx_artist is None and any(tok in hnorm for tok in candidates_artist):
            idx_artist = i

    out: List[ChartRow] = []
    for idx_row, data in enumerate(body, start=1):
        if not data or len(data) < 2:
            continue
        pos_text = data[idx_pos] if idx_pos is not None and idx_pos < len(data) else data[0]
        title_text = data[idx_title] if idx_title is not None and idx_title < len(data) else None
        artist_text = data[idx_artist] if idx_artist is not None and idx_artist < len(data) else None
        
        if title_text is None and len(data) >= 2:
            guess = data[1]
            if " - " in guess:
                t, _, a = guess.partition(" - ")
                title_text = title_text or t.strip()
                artist_text = artist_text or a.strip()
            elif " – " in guess:
                t, _, a = guess.partition(" – ")
                title_text = title_text or t.strip()
                artist_text = artist_text or a.strip()
            else:
                title_text = title_text or guess.strip()
        
        position = idx_row
        if pos_text and pos_text.isdigit():
             try:
                 position = int("".join(ch for ch in pos_text if ch.isdigit()))
             except:
                 pass
        
        out.append(
            ChartRow(
                source_domain=domain,
                chart=chart_name,
                region=region,
                date=None,
                position=position,
                title=title_text or "",
                artist=artist_text or "",
                url=None,
                scraped_at=ts,
            )
        )
    return out


DomainHandler = Callable[[str, str], List[ChartRow]]


def scrape_by_domain(url: str, chart_name: str) -> List[ChartRow]:
    domain = get_domain(url)
    if not domain:
        return []
    if domain == "turntablecharts.com":
        return scrape_turntablecharts(url, chart_name)
    if domain == "billboard.com":
        return scrape_billboard_greatest(url, chart_name)
    if "wikipedia.org" in domain:
        return scrape_wikipedia_list(url, chart_name)
        
    return generic_browser_chart(url, chart_name)


def rows_to_dataframe(rows: List[ChartRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "source_domain", "chart", "region", "date",
                "position", "title", "artist", "url", "scraped_at",
            ]
        )
    return pd.DataFrame([asdict(r) for r in rows])


def append_or_write(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, index=False)


def run_from_catalog(
    catalog_csv: Path,
    out_dir: Path,
    limit_per_chart: Optional[int] = None,
    only_domains: Optional[List[str]] = None,
    skip_domains: Optional[List[str]] = None,
) -> Tuple[int, int]:
    df = pd.read_csv(catalog_csv)
    if "official_site" not in df.columns:
        print(f"[catalog] Column 'official_site' not found in {catalog_csv}", file=sys.stderr)
        return (0, 0)
    sub = df.dropna(subset=["official_site"]).copy()
    total_rows = 0
    processed = 0
    for _, row in sub.iterrows():
        chart_name = str(row.get("chart") or "").strip() or "chart"
        site = str(row.get("official_site") or "").strip()
        if not site:
            continue
        domain = get_domain(site)
        if only_domains and domain not in only_domains:
            continue
        if skip_domains and domain in skip_domains:
            continue
        try:
            rows = scrape_by_domain(site, chart_name)
            if limit_per_chart is not None:
                rows = rows[:limit_per_chart]
            df_rows = rows_to_dataframe(rows)
            if not df_rows.empty:
                out_path = out_dir / f"{domain}.csv"
                append_or_write(df_rows, out_path)
                total_rows += len(df_rows)
                print(f"  -> Wrote {len(df_rows)} rows for {domain} / {chart_name}")
            else:
                print(f"  -> No rows found for {domain} / {chart_name}")
            processed += 1
        except Exception as e:
            print(f"[error] {domain} {chart_name}: {e}", file=sys.stderr)
        time.sleep(0.4)
    return (processed, total_rows)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape official chart sites from catalog CSV")
    p.add_argument("--catalog", type=str, default="data/chart_official_sites.csv", help="Path to chart_official_sites.csv")
    p.add_argument("--outdir", type=str, default="data/official_charts", help="Directory to write per-domain CSV files")
    p.add_argument("--limit", type=int, default=None, help="Limit rows per chart")
    p.add_argument("--only-domains", type=str, default="", help="Comma-separated domain whitelist")
    p.add_argument("--skip-domains", type=str, default="", help="Comma-separated domain blacklist")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    only = [d.strip().lower() for d in args.only_domains.split(",") if d.strip()] if args.only_domains else None
    skip = [d.strip().lower() for d in args.skip_domains.split(",") if d.strip()] if args.skip_domains else None
    processed, total = run_from_catalog(
        catalog_csv=Path(args.catalog),
        out_dir=Path(args.outdir),
        limit_per_chart=args.limit,
        only_domains=only,
        skip_domains=skip,
    )
    print(f"Processed {processed} sites. Wrote {total} rows to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
