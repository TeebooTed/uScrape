# uScrape

Keyword web-scraping desktop applet (Tk / ttkbootstrap).

Scrapes sites for configured keywords, with async HTTP (`aiohttp`), optional Playwright rendering, rate limiting, and CSV export. Bundled for packaging via PyInstaller (`uScrape.spec`).

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install
python uScrape.py
```

## Package

```bash
pyinstaller uScrape.spec
```

Source synced from local ThinkPad workspace (2026-10-06). Binary build artifacts and virtualenvs are intentionally not in this repo.
