"""
Collectors for popular music charts with a unified schema.

Outputs a CSV with columns:
    source,domain,chart,region,date,position,title,artist,url,scraped_at

Supported sources:
  - Billboard (via billboard.py)
  - Apple Music RSS (Top Songs by country)
  - Spotify Charts CSV (Top 200 by region)

Usage examples:
  # Quick pull for a few well-known charts
  python scraping/collect_songs.py --billboard "hot-100,global-200,global-excl-us" --apple "us,gb,ng" --spotify "global,us,gb" --limit 100 --out data/songs_latest.csv

  # Append to existing dataset
  python scraping/collect_songs.py --billboard hot-100 --date 2025-11-08 --append data/top_songs.csv
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional
import argparse
import csv
import json
import os
import sys
import time

import pandas as pd
import requests

# Optional import (only needed when using --billboard)
try:
    import billboard
except Exception:  # pragma: no cover - handled at runtime when feature used
    billboard = None

# Optional import (only needed when using --spotify-playlists)
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
except Exception:  # pragma: no cover
    spotipy = None
    SpotifyClientCredentials = None
    SpotifyOAuth = None

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)


@dataclass
class SongRow:
    source: str              # 'billboard' | 'apple_rss' | 'spotify_charts'
    domain: str              # e.g., 'billboard.com'
    chart: str               # chart identifier (e.g., 'Hot 100', 'Top Songs')
    region: Optional[str]    # ISO country code or 'global'
    date: Optional[str]      # chart date (YYYY-MM-DD) when applicable
    position: int            # 1-based rank
    title: str               # track title
    artist: str              # primary artist string
    url: Optional[str]       # canonical link to song or chart item
    scraped_at: str          # ISO timestamp


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -------------------- Billboard --------------------
def normalize_billboard_chart_name(name: str) -> str:
    return name.replace("_", "-").lower()


def collect_billboard(
    chart_names: Iterable[str],
    date: Optional[str],
    max_items: int | None = None,
) -> List[SongRow]:
    if billboard is None:
        raise RuntimeError(
            "billboard.py is not installed. Install with: pip install billboard.py"
        )
    out: List[SongRow] = []
    ts = utcnow_iso()
    for raw in chart_names:
        chart_id = normalize_billboard_chart_name(raw.strip())
        # billboard.ChartData accepts e.g. 'hot-100', 'billboard-200', 'global-200'
        data = billboard.ChartData(chart_id, date=date)
        # date string on object is chart week (YYYY-MM-DD)
        week = getattr(data, "date", None)
        # entries is a list of ChartEntry objects
        for i, e in enumerate(data):
            if max_items is not None and i >= max_items:
                break
            out.append(
                SongRow(
                    source="billboard",
                    domain="billboard.com",
                    chart=chart_id,
                    region=None if not chart_id.startswith("global") else "global",
                    date=week,
                    position=int(e.rank),
                    title=e.title or "",
                    artist=e.artist or "",
                    url=getattr(e, "spotifyLink", None) or getattr(e, "itunesLink", None),
                    scraped_at=ts,
                )
            )
        # be polite: Billboard source is rate-limited in the library implementation
        time.sleep(0.5)
    return out


# -------------------- Apple RSS --------------------
def apple_rss_top_songs(country_code: str, limit: int = 100) -> List[SongRow]:
    """
    Apple Music / iTunes RSS JSON (unauthenticated).
    Endpoint shape:
      https://rss.itunes.apple.com/api/v1/{cc}/itunes-music/top-songs/all/{limit}/explicit.json
    """
    def _fetch(url: str) -> dict:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                r = requests.get(url, headers=headers, timeout=30)
                # Some RSS endpoints return 503 intermittently; try again
                if r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                # Exponential backoff
                time.sleep(0.6 * (2**attempt))
        # If we exhaust retries, raise the last error
        if last_err:
            raise last_err
        return {}

    cc = country_code.lower()
    limit_int = int(limit)

    # Try explicit first; if it fails, fallback to 'non-explicit' feed
    primary = f"https://rss.itunes.apple.com/api/v1/{cc}/itunes-music/top-songs/all/{limit_int}/explicit.json"
    fallback = f"https://rss.itunes.apple.com/api/v1/{cc}/itunes-music/top-songs/all/{limit_int}/non-explicit.json"
    try:
        data = _fetch(primary)
    except Exception:
        data = _fetch(fallback)

    ts = utcnow_iso()
    # feed.updated is an ISO timestamp; author etc. included
    feed = data.get("feed", {})
    updated = feed.get("updated")  # e.g., '2025-11-09T10:00:00-07:00'

    items = []
    for pos, it in enumerate(feed.get("results", []), start=1):
        title = it.get("name") or ""
        artist = it.get("artistName") or ""
        url_item = it.get("url")
        items.append(
            SongRow(
                source="apple_rss",
                domain="rss.itunes.apple.com",
                chart="top-songs",
                region=country_code.lower(),
                date=updated[:10] if isinstance(updated, str) and len(updated) >= 10 else None,
                position=pos,
                title=title,
                artist=artist,
                url=url_item,
                scraped_at=ts,
            )
        )
    return items


def collect_apple(country_codes: Iterable[str], limit: int) -> List[SongRow]:
    out: List[SongRow] = []
    for cc in country_codes:
        try:
            out.extend(apple_rss_top_songs(cc.strip(), limit=limit))
        except Exception as e:  # pragma: no cover
            print(f"[apple] Skipping {cc}: {e}", file=sys.stderr)
        time.sleep(0.3)
    return out


# -------------------- Spotify Charts (CSV) --------------------
def spotify_charts_csv_url(region: str, date: str = "latest", frequency: str = "daily") -> str:
    """
    Builds a direct CSV download URL.
    Examples:
      - latest daily global:  /regional/global/daily/latest/download
      - latest weekly US:     /regional/us/weekly/latest/download
    """
    region = region.lower()
    frequency = frequency.lower()
    return f"https://spotifycharts.com/regional/{region}/{frequency}/{date}/download"


def spotify_charts_browser_csv(region: str, date: str = "latest", frequency: str = "daily") -> str:
    """
    Drive the official 'Download to CSV' via headless Chromium and return CSV text.
    """
    from playwright.sync_api import sync_playwright
    from io import StringIO

    region = region.lower()
    frequency = frequency.lower()
    page_url = f"https://spotifycharts.com/regional/{region}/{frequency}/{date}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        # accept_downloads=True is crucial so we can grab the CSV bytes
        context = browser.new_context(user_agent=USER_AGENT, accept_downloads=True, locale="en-US")
        page = context.new_page()
        page.set_default_timeout(30000)
        page.goto(page_url, wait_until="domcontentloaded")

        # The site has used different selectors over time; try several:
        # 1) a[href*="/download"] in header
        # 2) explicit text 'Download to CSV'
        # 3) a.header-csv (older)
        # Wait for any candidate to appear
        sel_candidates = [
            "a[href*='/download']",
            "a:has-text('Download to CSV')",
            "a.header-csv",
            "a.header__download",  # legacy
        ]

        found = None
        for css in sel_candidates:
            if page.locator(css).count():
                found = css
                break
            try:
                page.wait_for_selector(css, timeout=4000)
                found = css
                break
            except Exception:
                pass

        if not found:
            # As a fallback, the direct download URL sometimes works once cookies are set
            dl_url = f"https://spotifycharts.com/regional/{region}/{frequency}/{date}/download"
            resp = page.request.get(dl_url)
            if resp.ok:
                csv_text = resp.text()
                browser.close()
                return csv_text
            raise RuntimeError("CSV download button not found on page; site structure may have changed.")

        with page.expect_download() as dl_info:
            page.locator(found).first.click()

        download = dl_info.value
        csv_bytes = download.bytes()
        browser.close()
        return csv_bytes.decode("utf-8", errors="replace")


def collect_spotify(
    regions: Iterable[str],
    date: str = "latest",
    frequency: str = "daily",
    force_browser: bool = False,
) -> List[SongRow]:
    headers = {"User-Agent": USER_AGENT, "Referer": "https://spotifycharts.com"}
    out: List[SongRow] = []
    ts = utcnow_iso()
    for rgn in regions:
        rgn = rgn.lower().strip()
        if not rgn:
            continue
        csv_text = None
        
        # 1) Fast path: direct GET, unless the user forced browser mode
        if not force_browser:
            try:
                url = spotify_charts_csv_url(rgn, date=date, frequency=frequency)
                resp = requests.get(url, headers=headers, timeout=60)
                resp.raise_for_status()
                txt = resp.text.strip()
                looks_like_html = txt.lower().startswith("<!doctype") or txt.lower().startswith("<html")
                too_short = len(txt) < 100
                has_header = "Position,Track Name" in txt
                if not looks_like_html and not too_short and has_header:
                    csv_text = resp.text
                else:
                    print(f"[spotify] Info: {rgn} direct GET returned HTML/short. Falling back to headless download.", file=sys.stderr)
            except Exception as e:
                print(f"[spotify] Info: {rgn} direct GET failed ({e}). Falling back to headless download.", file=sys.stderr)
        
        # 2) Browser fallback (or forced)
        if csv_text is None:
            try:
                csv_text = spotify_charts_browser_csv(rgn, date=date, frequency=frequency)
            except Exception as e:
                print(f"[spotify] Skipping {rgn}: headless download failed ({e})", file=sys.stderr)
                time.sleep(0.6)
                continue
        
        # 3) Parse CSV
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(csv_text), skiprows=1)
            if df.empty:
                print(f"[spotify] Warning: {rgn} CSV parsed but is empty", file=sys.stderr)
                continue
            
            # Expected columns: Position, Track Name, Artist, Streams, URL
            for _, row in df.iterrows():
                pos = int(row.get("Position"))
                title = str(row.get("Track Name", "") or "")
                artist = str(row.get("Artist", "") or "")
                url_item = str(row.get("URL") or "")
                out.append(
                    SongRow(
                        source="spotify_charts",
                        domain="spotifycharts.com",
                        chart=f"top-200-{frequency}",
                        region=rgn,
                        date=None if date == "latest" else date,
                        position=pos,
                        title=title,
                        artist=artist,
                        url=url_item,
                        scraped_at=ts,
                    )
                )
        except Exception as e:
            print(f"[spotify] Skipping {rgn}: parse error ({e})", file=sys.stderr)
        time.sleep(0.4)
    return out


# -------------------- I/O helpers --------------------
def rows_to_dataframe(rows: List[SongRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])


def append_or_write(df: pd.DataFrame, out_path: Path, append: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if append and out_path.exists():
        # Append without headers if file exists
        df.to_csv(out_path, mode="a", header=False, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)


# -------------------- CLI --------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect songs from popular charts")
    p.add_argument("--billboard", type=str, default="", help="Comma-separated chart ids (e.g., hot-100,billboard-200,global-200)")
    p.add_argument("--date", type=str, default=None, help="Billboard chart date (YYYY-MM-DD)")
    p.add_argument("--apple", type=str, default="", help="Comma-separated country codes for Apple RSS (e.g., us,gb,ng)")
    p.add_argument("--spotify", type=str, default="", help="Comma-separated regions for Spotify charts (e.g., global,us,gb)")
    p.add_argument("--spotify-regions-file", type=str, default="", help="Path to text file with one region code per line (e.g., regions.txt)")
    p.add_argument("--spotify-browser", action="store_true", help="Force headless browser flow for Spotify Charts CSV")
    p.add_argument("--spotify-date", type=str, default="latest", help="Date for Spotify charts (YYYY-MM-DD or 'latest')")
    p.add_argument("--spotify-frequency", type=str, default="daily", choices=["daily", "weekly"], help="Spotify charts frequency")
    p.add_argument("--spotify-playlists", type=str, default="", help="Comma-separated Spotify playlist IDs or URLs (e.g., 37i9dQZEVXbMDoHDwVN2tF)")
    p.add_argument("--spotify-user-auth", action="store_true", help="Use user OAuth instead of client credentials (required for editorial playlists)")
    p.add_argument("--limit", type=int, default=100, help="Max items per chart where applicable")
    p.add_argument("--out", type=str, default="data/songs_latest.csv", help="Output CSV")
    p.add_argument("--append", type=str, default="", help="Append to this CSV path instead of --out")
    return p.parse_args(argv)


def parse_playlist_id(s: str) -> str:
    # Accept raw IDs or full URLs
    if "open.spotify.com/playlist/" in s:
        s = s.split("playlist/")[1]
    if "?" in s:
        s = s.split("?")[0]
    return s.strip()


def is_spotify_editorial(pid: str) -> bool:
    """
    Detect Spotify editorial/algorithmic playlists.
    These typically start with '37i9dQZF' and are restricted from client-credentials access.
    """
    return pid.startswith("37i9dQZF") or pid.startswith("37i9dQZEVXb")


def read_regions_file(path: str) -> List[str]:
    """Read region codes from a text file (one per line, comments with #)."""
    path = path.strip()
    regs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            regs.append(line.split()[0].lower())
    return regs


def collect_spotify_playlists(playlist_ids: Iterable[str], use_user_auth: bool = False) -> List[SongRow]:
    if spotipy is None or SpotifyClientCredentials is None:
        raise RuntimeError("spotipy is not installed. Install with: pip install spotipy")
    
    # Choose authentication method
    if use_user_auth:
        if SpotifyOAuth is None:
            raise RuntimeError(
                "User OAuth requires spotipy. Install with: pip install spotipy\n"
                "Also set environment variables:\n"
                "  SPOTIPY_CLIENT_ID=your_client_id\n"
                "  SPOTIPY_CLIENT_SECRET=your_client_secret\n"
                "  SPOTIPY_REDIRECT_URI=http://localhost:8888/callback (or your redirect URI)"
            )
        try:
            auth = SpotifyOAuth(scope="playlist-read-private playlist-read-collaborative")
            sp = spotipy.Spotify(auth_manager=auth)
        except Exception as e:
            if "client_id" in str(e).lower() or "SPOTIPY_CLIENT_ID" in str(e):
                raise RuntimeError(
                    "Spotify API credentials required for user OAuth. Set environment variables:\n"
                    "  SPOTIPY_CLIENT_ID=your_client_id\n"
                    "  SPOTIPY_CLIENT_SECRET=your_client_secret\n"
                    "  SPOTIPY_REDIRECT_URI=http://localhost:8888/callback\n"
                    "Get credentials from: https://developer.spotify.com/dashboard"
                ) from e
            raise
    else:
        try:
            auth = SpotifyClientCredentials()
            sp = spotipy.Spotify(auth_manager=auth)
        except Exception as e:
            if "client_id" in str(e).lower() or "SPOTIPY_CLIENT_ID" in str(e):
                raise RuntimeError(
                    "Spotify API credentials required. Set environment variables:\n"
                    "  SPOTIPY_CLIENT_ID=your_client_id\n"
                    "  SPOTIPY_CLIENT_SECRET=your_client_secret\n"
                    "Get credentials from: https://developer.spotify.com/dashboard"
                ) from e
            raise
    ts = utcnow_iso()
    out: List[SongRow] = []
    for raw in playlist_ids:
        pid = parse_playlist_id(raw)
        
        # Check if this is a Spotify editorial playlist
        if is_spotify_editorial(pid) and not use_user_auth:
            print(
                f"[spotify-api] Warning: Playlist {pid} appears to be a Spotify editorial playlist. "
                "Client-credentials requests now return 404 due to API restrictions (late 2024 policy change). "
                "Use --spotify-user-auth flag and copy the playlist to a user-owned playlist, "
                "or use Billboard/Apple RSS for chart data instead.",
                file=sys.stderr
            )
        
        try:
            # Fetch playlist metadata (add market parameter for region-restricted playlists)
            try:
                meta = sp.playlist(pid, fields="name,external_urls.spotify", market="US")
            except Exception:
                # Try without market parameter if US market fails
                meta = sp.playlist(pid, fields="name,external_urls.spotify")
            pname = meta.get("name") or pid
            purl = meta.get("external_urls", {}).get("spotify")

            # Paginated items
            offset = 0
            limit = 100
            position = 0
            while True:
                try:
                    items = sp.playlist_items(
                        pid,
                        offset=offset,
                        limit=limit,
                        fields="items(track(name,artists(name),external_urls.spotify)),total,next",
                        market="US",
                    )
                except Exception:
                    # Try without market parameter if US market fails
                    items = sp.playlist_items(
                        pid,
                        offset=offset,
                        limit=limit,
                        fields="items(track(name,artists(name),external_urls.spotify)),total,next",
                    )
                tracks = items.get("items", [])
                if not tracks:
                    break
                for it in tracks:
                    track = it.get("track") or {}
                    title = track.get("name") or ""
                    artists = track.get("artists") or []
                    artist = ", ".join(a.get("name") for a in artists if a and a.get("name"))
                    url_item = (track.get("external_urls") or {}).get("spotify")
                    position += 1
                    out.append(
                        SongRow(
                            source="spotify_api",
                            domain="api.spotify.com",
                            chart=f"playlist:{pname}",
                            region=None,
                            date=None,
                            position=position,
                            title=title,
                            artist=artist,
                            url=url_item or purl,
                            scraped_at=ts,
                        )
                    )
                offset += len(tracks)
                if not items.get("next"):
                    break
        except Exception as e:  # pragma: no cover
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                if is_spotify_editorial(pid) and not use_user_auth:
                    print(
                        f"[spotify-api] Playlist {pid} returned 404. "
                        "This is a Spotify editorial playlist and requires user OAuth (not client credentials). "
                        "Due to late 2024 API restrictions, editorial playlists return 404 with client-credentials tokens. "
                        "Use --spotify-user-auth flag and copy the playlist to a user-owned playlist, "
                        "or use Billboard/Apple RSS for chart data instead.",
                        file=sys.stderr
                    )
                else:
                    print(
                        f"[spotify-api] Playlist {pid} not found. "
                        "It may be private, deleted, or the ID is invalid. "
                        "Make sure the playlist is public and the ID is correct.",
                        file=sys.stderr
                    )
            else:
                print(f"[spotify-api] Skipping {pid}: {e}", file=sys.stderr)
        time.sleep(0.2)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    all_rows: List[SongRow] = []

    # Billboard
    if args.billboard:
        charts = [c.strip() for c in args.billboard.split(",") if c.strip()]
        try:
            all_rows.extend(collect_billboard(charts, date=args.date, max_items=args.limit))
        except Exception as e:  # pragma: no cover
            print(f"[billboard] {e}", file=sys.stderr)

    # Apple RSS
    if args.apple:
        country_codes = [cc.strip() for cc in args.apple.split(",") if cc.strip()]
        all_rows.extend(collect_apple(country_codes, limit=args.limit))

    # Spotify charts
    if args.spotify or args.spotify_regions_file:
        regions = []
        if args.spotify:
            regions.extend([r.strip() for r in args.spotify.split(",") if r.strip()])
        if args.spotify_regions_file:
            try:
                regions.extend(read_regions_file(args.spotify_regions_file))
            except Exception as e:
                print(f"[spotify] Regions file error: {e}", file=sys.stderr)
        # de-dup while keeping order
        seen = set()
        regions = [r for r in regions if not (r in seen or seen.add(r))]
        if regions:
            all_rows.extend(
                collect_spotify(
                    regions,
                    date=args.spotify_date,
                    frequency=args.spotify_frequency,
                    force_browser=args.spotify_browser,
                )
            )

    # Spotify via Web API playlists
    if args.spotify_playlists:
        pids = [p.strip() for p in args.spotify_playlists.split(",") if p.strip()]
        all_rows.extend(collect_spotify_playlists(pids, use_user_auth=args.spotify_user_auth))

    if not all_rows:
        print("No rows collected. Provide at least one source flag.", file=sys.stderr)
        return 2

    df = rows_to_dataframe(all_rows)

    out_path = Path(args.append or args.out)
    append = bool(args.append)
    append_or_write(df, out_path, append=append)
    print(f"Saved {len(df):,} rows to {out_path} (append={append})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


