import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.toast import ToastNotification
from ttkbootstrap.tooltip import ToolTip
import asyncio
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import csv
import json
import time
import logging
from ratelimit import limits, sleep_and_retry
from ratelimit.exception import RateLimitException
import re
from datetime import datetime, timezone, timedelta
from playwright.async_api import async_playwright
import os
from transformers import pipeline

# Initialize logging
logging.basicConfig(filename='scrape_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use 'vapor' theme for a modern look
root = ttk.Window("uScrape v .90", themename="vapor")
root.geometry("900x700")

# Style adjustments
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TEntry", font=("Helvetica", 11), padding=10)
style.configure("TButton", font=("Helvetica", 10), padding=10)

# Global variables
pause_flag = False
cancel_flag = False
last_processed_url = None
export_frame = None
scrape_results = []  # Store scrape results globally
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Global summarizer for performance

# Timezone for PST
PST = timezone(-timedelta(hours=8))

CACHE_FILE = 'page_count_cache.json'
CACHE_MAX_SIZE = 100
CACHE_EXPIRATION = timedelta(days=7)

def cache_cleanup():
    if not os.path.exists(CACHE_FILE):
        return
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        now = datetime.now(timezone.utc)
        cleaned_cache = {}
        count = 0
        for url, data in sorted(cache.items(), key=lambda item: item[1]['timestamp'], reverse=True):
            cache_time = datetime.fromisoformat(data['timestamp'])
            if now - cache_time < CACHE_EXPIRATION:
                cleaned_cache[url] = data
                count += 1
                if count >= CACHE_MAX_SIZE:
                    break
        with open(CACHE_FILE, 'w') as f:
            json.dump(cleaned_cache, f)
        logging.info(f"Cache cleaned: {len(cleaned_cache)} entries retained.")
    except Exception as e:
        logging.error(f"Error during cache cleanup: {e}")

def save_cache(url, count):
    cache_cleanup()
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}
        cache[url] = {"count": count, "timestamp": datetime.now(timezone.utc).isoformat()}
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        logging.error(f"Error saving cache for {url}: {e}")

def load_cache(url):
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
            if url in cache:
                cached_data = cache[url]
                cached_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now(timezone.utc) - cached_time < timedelta(hours=24):
                    return cached_data['count']
        except json.JSONDecodeError:
            logging.error("Cache file corrupted, ignoring.")
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
    return None

def normalize_url(url):
    try:
        url = url.strip()
        parsed = urlparse(url)
        if not parsed.scheme:
            url = 'https://' + url if url.startswith('www.') else 'https://' + url
        url = url.rstrip('/')
        return url
    except Exception as e:
        logging.error(f"Error normalizing URL: {e}")
        return url

def match_keywords(text, keywords):
    try:
        text = text.lower()
        results = {}
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = re.findall(pattern, text)
            if matches:
                results[keyword] = len(matches)
        return results
    except Exception as e:
        logging.error(f"Error matching keywords: {e}")
        return {}

@sleep_and_retry
@limits(calls=20, period=1)
async def fetch(session, url):
    if cancel_flag:
        return None
    max_retries = 5
    retries = 0
    while retries < max_retries:
        if cancel_flag:
            return None
        try:
            async with session.get(url, timeout=10) as response:
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' in content_type:
                    return await response.text()
                else:
                    logging.info(f"Skipping non-HTML content for {url}: {content_type}")
                    return None
        except RateLimitException as e:
            retries += 1
            if retries == max_retries or cancel_flag:
                logging.error(f"Failed to fetch {url} after {max_retries} retries: {e}")
                return None
            sleep_time = e.period_remaining + 0.1
            logging.warning(f"Rate limit hit for {url}. Retrying in {sleep_time} seconds.")
            await asyncio.sleep(sleep_time)
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            if not url.startswith("https"):
                url = url.replace("http://", "https://", 1)
                try:
                    async with session.get(url, timeout=10) as response:
                        content_type = response.headers.get('Content-Type', '').lower()
                        if 'text/html' in content_type:
                            return await response.text()
                        else:
                            logging.info(f"Skipping non-HTML content for {url}: {content_type}")
                            return None
                except Exception as e:
                    logging.error(f"Error fetching {url} with HTTPS: {e}")
                    return None
    return None

async def fetch_dynamic(url):
    if cancel_flag:
        return None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url)
            content = await page.content()
            await browser.close()
            return content
    except Exception as e:
        logging.error(f"Error fetching dynamic content from {url}: {e}")
        return None

def parse_page_response(response):
    try:
        soup = BeautifulSoup(response, 'lxml')
        links = []
        for link in soup.find_all(['a', 'link'], href=True):
            href = link['href']
            if not any(href.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.js', '.css']):
                links.append(href)
        logging.info(f"Found {len(links)} HTML links on page")
        return links
    except Exception as e:
        logging.error(f"Error parsing page response: {e}")
        return []

async def count_pages_async(url, batch_size=10, max_depth=10, max_pages=1000):
    if cancel_flag:
        return 0
    visited = set()
    queue = [(url, 0)]
    count = 0
    
    def is_unique_url(page_url, visited):
        normalized = normalize_url(page_url)
        return normalized not in visited

    async with ClientSession() as session:
        while queue and count < max_pages and not cancel_flag:
            tasks = []
            batch = queue[:batch_size]
            queue = queue[batch_size:]
            
            for current_url, current_depth in batch:
                if is_unique_url(current_url, visited) and current_depth <= max_depth:
                    tasks.append(fetch(session, current_url))
            
            responses = await asyncio.gather(*tasks)
            
            for i, response in enumerate(responses):
                if cancel_flag:
                    return count
                current_url, current_depth = batch[i]
                update_status(f"Counting pages: Processing {current_url}")
                if response is None:
                    continue
                visited.add(normalize_url(current_url))
                count += 1
                links = parse_page_response(response)
                for link in links:
                    next_link = urljoin(url, link)
                    if is_unique_url(next_link, visited) and next_link.startswith(url):
                        queue.append((next_link, current_depth + 1))
    
    return count

async def count_pages_with_cache(url, batch_size=10, max_depth=10, max_pages=1000):
    cached_count = load_cache(url)
    if cached_count is not None:
        update_status(f"Using cached page count for {url}")
        return cached_count
    try:
        count = await count_pages_async(url, batch_size, max_depth, max_pages)
        save_cache(url, count)
        return count
    except Exception as e:
        logging.error(f"Error counting pages for {url}: {e}")
        return 0

async def scrape_async(session, url, keywords, batch_size=10, max_depth=10, max_pages=1000):
    global scrape_results
    if cancel_flag:
        return []
    scrape_results = []
    visited = set()
    queue = [url]
    current_page = 0
    actual_pages_processed = 0

    update_status("Starting page count...")
    page_count = await count_pages_with_cache(url, batch_size, max_depth, max_pages)
    update_gui(f"Total pages found: {page_count}\n\n")
    
    update_gui("{:<20} {:<80} {:<10}\n".format("Keyword", "Page", "Count"))
    update_gui("-" * 110 + "\n")

    while queue and current_page < page_count and not cancel_flag:
        if pause_flag:
            last_processed_url = queue[0]
            await asyncio.sleep(0.1)
            continue
        current_url = queue.pop(0)
        if normalize_url(current_url) not in visited:
            visited.add(normalize_url(current_url))
            actual_pages_processed += 1
            current_page += 1
            update_status(f"Scraping: {current_url}")
            update_progress(current_page, page_count)
            
            try:
                if 'javascript' in current_url.lower():
                    content = await fetch_dynamic(current_url)
                else:
                    content = await fetch(session, current_url)
                
                if content is None or cancel_flag:
                    continue

                soup = BeautifulSoup(content, 'lxml')
                text = soup.get_text()
                keyword_counts = match_keywords(text, keywords)
                
                if keyword_counts:
                    for keyword, count in keyword_counts.items():
                        new_line = "{:<20} {:<80} {:<10}\n".format(keyword, current_url, count)
                        update_gui(new_line)
                        scrape_results.append({"keyword": keyword, "page": current_url, "count": count})
                
                links = parse_page_response(content)
                for link in links:
                    next_link = urljoin(url, link)
                    if normalize_url(next_link) not in visited and next_link.startswith(url):
                        queue.append(next_link)
                        logging.info(f"Added to queue: {next_link}")
            except RateLimitException:
                update_status(f"Rate limit hit. Waiting before retrying...")
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"Error scraping {current_url}: {e}")

    if cancel_flag:
        update_status("Scraping cancelled")
        logging.info("Scraping operation was cancelled")
    else:
        update_status("Scraping completed")
    logging.info(f"Visited {len(visited)} unique URLs")
    
    if not scrape_results:
        update_gui("No keywords found across the website.")
    else:
        global export_frame
        if export_frame is None:
            export_frame = ttk.Frame(root)
            export_frame.pack(pady=10)
        else:
            for widget in export_frame.winfo_children():
                widget.destroy()
        
        ttk.Button(export_frame, text="Save as CSV", bootstyle="success", command=lambda: save_to_csv(scrape_results)).pack(side=LEFT, padx=5)
        ttk.Button(export_frame, text="Save as JSON", bootstyle="info", command=lambda: save_to_json(scrape_results)).pack(side=LEFT, padx=5)
        ttk.Button(export_frame, text="Summarize", bootstyle="primary", command=summarize_saturation).pack(side=LEFT, padx=5)
        ttk.Label(export_frame, text="Summary Focus:").pack(side=LEFT, padx=5)
        summary_focus = ttk.Combobox(export_frame, values=["Balanced", "Frequency", "Coverage"], state="readonly", width=10)
        summary_focus.pack(side=LEFT, padx=5)
        summary_focus.set("Balanced")  # Default value
    
    logging.info(f"Total pages processed: {actual_pages_processed}, Estimated pages: {page_count}")
    return scrape_results

def save_to_csv(data):
    try:
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['keyword', 'page', 'count']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
            messagebox.showinfo("Success", f"Results saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")
        messagebox.showerror("Error", f"Failed to save CSV file: {e}")

def save_to_json(data):
    try:
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("Success", f"Results saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving to JSON: {e}")
        messagebox.showerror("Error", f"Failed to save JSON file: {e}")

def analyze_saturation(results):
    if not results:
        return {"total_pages": 0, "keywords": {}}
    keyword_stats = {}
    pages_with_keywords = set()
    total_pages = len({r["page"] for r in results})
    for result in results:
        keyword = result["keyword"]
        page = result["page"]
        count = result["count"]
        if keyword not in keyword_stats:
            keyword_stats[keyword] = {"total_count": 0, "pages": set()}
        keyword_stats[keyword]["total_count"] += count
        keyword_stats[keyword]["pages"].add(page)
        pages_with_keywords.add(page)
    saturation_data = {
        "total_pages": total_pages,
        "pages_with_keywords": len(pages_with_keywords),
        "keywords": {}
    }
    for keyword, stats in keyword_stats.items():
        saturation_data["keywords"][keyword] = {
            "total_count": stats["total_count"],
            "page_coverage": len(stats["pages"]) / total_pages if total_pages > 0 else 0,
            "avg_count_per_page": stats["total_count"] / len(stats["pages"]) if stats["pages"] else 0
        }
    return saturation_data

def summarize_saturation():
    global scrape_results
    if not scrape_results:
        ToastNotification(title="No Data", message="Please scrape a website first.", duration=3000, bootstyle="warning").show()
        return
    
    saturation_data = analyze_saturation(scrape_results)
    if saturation_data["total_pages"] == 0:
        update_gui("\nNo data available to summarize.\n")
        return
    
    focus = export_frame.winfo_children()[-1].get()  # Get dropdown value
    text = f"The website has {saturation_data['total_pages']} unique pages scraped. "
    text += f"Keywords were found on {saturation_data['pages_with_keywords']} pages. "
    
    if focus == "Frequency":
        text += "Focus is on frequency: "
        for keyword, stats in saturation_data["keywords"].items():
            text += f"'{keyword}' appears {stats['total_count']} times, averaging {stats['avg_count_per_page']:.1f} per page it’s on. "
    elif focus == "Coverage":
        text += "Focus is on coverage: "
        for keyword, stats in saturation_data["keywords"].items():
            text += f"'{keyword}' covers {int(stats['page_coverage'] * 100)}% of pages. "
    else:  # Balanced
        text += "Balanced analysis: "
        for keyword, stats in saturation_data["keywords"].items():
            text += f"'{keyword}' appears {stats['total_count']} times across {int(stats['page_coverage'] * 100)}% of pages, averaging {stats['avg_count_per_page']:.1f} per page. "
    
    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
    except Exception as e:
        logging.error(f"Error generating AI summary: {e}")
        summary = text
        ToastNotification(title="Summary Error", message="AI summarization failed; raw text used.", duration=3000, bootstyle="danger").show()
    
    update_gui("\nWebsite Saturation Summary:\n")
    update_gui(f"{summary}\n")
    logging.info(f"Saturation summary generated: {summary}")

def update_gui(new_line):
    try:
        result_text.insert(ttk.END, new_line)
        result_text.see(ttk.END)
        root.update_idletasks()
    except Exception as e:
        logging.error(f"Error updating GUI: {e}")

def update_status(text):
    try:
        status_label.config(text=text)
        root.update_idletasks()
        logging.info(text)
    except Exception as e:
        logging.error(f"Error updating status: {e}")

def update_progress(current, total):
    try:
        progress = min((current / total) * 100 if total > 0 else 0, 100)
        progress_bar['value'] = progress
        progress_label.config(text=f"{int(progress)}% Complete")
        root.update_idletasks()
    except Exception as e:
        logging.error(f"Error updating progress: {e}")

def toggle_pause():
    global pause_flag
    pause_flag = not pause_flag
    pause_button.config(text="Resume" if pause_flag else "Pause")
    if not pause_flag and last_processed_url:
        queue.insert(0, last_processed_url)

def cancel_scrape():
    global cancel_flag
    cancel_flag = True
    update_status("Scraping operation cancelled")
    messagebox.showinfo("Cancelled", "Scraping has been cancelled.")
    reset_fields()

def reset_fields():
    global cancel_flag
    cancel_flag = False
    url_entry.delete(0, ttk.END)
    keywords_entry.delete(0, ttk.END)
    depth_entry.delete(0, ttk.END)
    depth_entry.insert(0, '10')
    pages_entry.delete(0, ttk.END)
    pages_entry.insert(0, '1000')
    rate_limit_entry.delete(0, ttk.END)
    rate_limit_entry.insert(0, '20')
    result_text.delete('1.0', ttk.END)
    status_label.config(text="Idle")
    progress_bar['value'] = 0
    progress_label.config(text="0% Complete")
    global pause_flag, last_processed_url, export_frame
    pause_flag = False
    last_processed_url = None
    if export_frame:
        for widget in export_frame.winfo_children():
            widget.destroy()
    pause_button.config(text="Pause")

async def main_async():
    global cancel_flag
    cancel_flag = False
    raw_url = url_entry.get().strip()
    if not raw_url:
        messagebox.showerror("Error", "Please enter a URL")
        return
    url = normalize_url(raw_url)
    keywords = [kw.strip() for kw in keywords_entry.get().split(',') if kw.strip()]
    if not keywords:
        messagebox.showerror("Error", "Please enter keywords")
        return
    try:
        max_depth = max(1, int(depth_entry.get() or 10))
        max_pages = max(1, int(pages_entry.get() or 1000))
        rate_limit = max(1, int(rate_limit_entry.get() or 20))
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for depth, pages, and rate limit.")
        return
    now = datetime.now(PST).strftime("%I:%M %p on %B %d, %Y PST")
    update_status(f"Starting scrape at {now}")
    try:
        async with ClientSession() as session:
            await scrape_async(session, url, keywords, max_depth=max_depth, max_pages=max_pages)
    except Exception as e:
        logging.error(f"Error in main async function: {e}")
        messagebox.showerror("Scraping Error", "An error occurred during scraping.")

def scrape_website():
    try:
        asyncio.run(main_async())
    except Exception as e:
        logging.error(f"Error running async main function: {e}")
        messagebox.showerror("Execution Error", "An error occurred while running the async function.")

def view_log():
    log_path = 'scrape_log.txt'
    if not os.path.exists(log_path):
        messagebox.showinfo("Log File", "No log file exists.")
    else:
        try:
            log_viewer = tk.Toplevel(root)
            log_viewer.title("Log Viewer")
            log_text = ttk.ScrolledText(log_viewer, width=100, height=30)
            log_text.pack(fill=tk.BOTH, expand=True)
            with open(log_path, 'r') as log:
                log_contents = log.read()
            log_text.insert(tk.END, log_contents)
        except Exception as e:
            logging.error(f"Error viewing log: {e}")
            messagebox.showerror("Error", "Could not open log file.")

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

# Create File menu
menu_bar = tk.Menu(root)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="View Log", command=view_log)
file_menu.add_separator()
file_menu.add_command(label="Close", command=on_closing)
menu_bar.add_cascade(label="File", menu=file_menu)
root.config(menu=menu_bar)

# Main Frame
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

# Input Frame
input_frame = ttk.Frame(main_frame, padding=10)
input_frame.pack(fill=tk.X, pady=(0, 20))

ttk.Label(input_frame, text="Enter Base URL:", bootstyle=INFO).grid(row=0, column=0, padx=(0, 10), sticky="w")
url_entry = ttk.Entry(input_frame, width=50)
url_entry.grid(row=0, column=1, sticky="ew")
ToolTip(url_entry, text="Enter the base URL (e.g., example.com)")

ttk.Label(input_frame, text="Keywords (comma-separated):", bootstyle=INFO).grid(row=1, column=0, padx=(0, 10), sticky="w")
keywords_entry = ttk.Entry(input_frame, width=50)
keywords_entry.grid(row=1, column=1, sticky="ew")
ToolTip(keywords_entry, text="Enter keywords separated by commas (e.g., school, education)")

# Settings Frame
settings_frame = ttk.Frame(main_frame, padding=10)
settings_frame.pack(fill=tk.X, pady=(0, 20))

ttk.Label(settings_frame, text="Max Depth:", bootstyle=INFO).grid(row=0, column=0, padx=(0, 10), sticky="w")
depth_entry = ttk.Entry(settings_frame, width=5)
depth_entry.grid(row=0, column=1, padx=(0, 20))
depth_entry.insert(0, '10')
ToolTip(depth_entry, text="Maximum depth of links to follow")

ttk.Label(settings_frame, text="Max Pages:", bootstyle=INFO).grid(row=0, column=2, padx=(0, 10), sticky="w")
pages_entry = ttk.Entry(settings_frame, width=5)
pages_entry.grid(row=0, column=3, padx=(0, 20))
pages_entry.insert(0, '1000')
ToolTip(pages_entry, text="Maximum number of pages to scrape")

ttk.Label(settings_frame, text="API Calls/sec:", bootstyle=INFO).grid(row=0, column=4, padx=(0, 10), sticky="w")
rate_limit_entry = ttk.Entry(settings_frame, width=5)
rate_limit_entry.grid(row=0, column=5)
rate_limit_entry.insert(0, '20')
ToolTip(rate_limit_entry, text="API calls per second")

# Actions Frame
action_frame = ttk.Frame(main_frame, padding=10)
action_frame.pack(fill=tk.X, pady=(0, 20))

scrape_button = ttk.Button(action_frame, text="Scrape Website", bootstyle=(PRIMARY, OUTLINE), command=scrape_website)
scrape_button.pack(side=LEFT, padx=5)
ToolTip(scrape_button, text="Start scraping the website")

pause_button = ttk.Button(action_frame, text="Pause", bootstyle=(SECONDARY, OUTLINE), command=toggle_pause)
pause_button.pack(side=LEFT, padx=5)
ToolTip(pause_button, text="Pause or resume scraping")

cancel_button = ttk.Button(action_frame, text="Cancel", bootstyle=(DANGER, OUTLINE), command=cancel_scrape)
cancel_button.pack(side=LEFT, padx=5)
ToolTip(cancel_button, text="Cancel the current scraping operation")

reset_button = ttk.Button(action_frame, text="Reset", bootstyle=(WARNING, OUTLINE), command=reset_fields)
reset_button.pack(side=LEFT, padx=5)
ToolTip(reset_button, text="Reset all input fields")

# Results Frame
results_frame = ttk.Frame(main_frame)
results_frame.pack(fill=tk.BOTH, expand=True)

result_text = ttk.ScrolledText(results_frame, height=15, width=100, font=("Helvetica", 10), wrap=tk.WORD)
result_text.pack(fill=tk.BOTH, expand=True)

# Status and Progress
status_frame = ttk.Frame(main_frame, padding=10)
status_frame.pack(fill=tk.X, side=tk.BOTTOM)

status_label = ttk.Label(status_frame, text="Idle", font=("Helvetica", 12), bootstyle=INFO)
status_label.pack(side=tk.LEFT, padx=(0, 20))

progress_bar = ttk.Progressbar(status_frame, orient=HORIZONTAL, length=300, mode='determinate', bootstyle=(SUCCESS, STRIPED))
progress_bar.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

progress_label = ttk.Label(status_frame, text="0% Complete", font=("Helvetica", 12))
progress_label.pack(side=tk.LEFT)

# Start the GUI
root.mainloop()