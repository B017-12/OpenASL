#!/usr/bin/env python3
"""
Fetch public-domain ASL manual alphabet SVG illustrations (A-Z) from Wikimedia Commons.
Source: https://commons.wikimedia.org/wiki/Category:ASL_letters (wpclipart.com, Public Domain).
No AI-generated imagery is used.
"""

import json
import pathlib
import string
import time
import urllib.parse
import urllib.request

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "public" / "assets" / "letters"
USER_AGENT = "OpenASL/0.1 (https://github.com/B017-12/OpenASL)"


def fetch_svg_url(letter: str) -> str:
    title = f"File:Sign_language_{letter}.svg"
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        }
    )
    api_url = f"https://commons.wikimedia.org/w/api.php?{query}"
    req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    pages = payload["query"]["pages"]
    page = next(iter(pages.values()))
    return page["imageinfo"][0]["url"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for letter in string.ascii_uppercase:
        dest = OUT_DIR / f"{letter}.svg"
        if dest.exists() and dest.stat().st_size > 200:
            print(f"[skip] {dest.name} already present")
            continue
        url = fetch_svg_url(letter)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            svg_bytes = resp.read()
        if b"<svg" not in svg_bytes.lower():
            raise RuntimeError(f"Unexpected content for {letter}: not an SVG")
        dest.write_bytes(svg_bytes)
        print(f"[ok]   {dest.name} ({len(svg_bytes)} bytes)")
        time.sleep(0.15)


if __name__ == "__main__":
    main()
