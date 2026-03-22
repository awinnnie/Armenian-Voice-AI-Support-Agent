"""
ACBA Bank scraper.
 
Scrapes loan products, deposit products, and branch information from
https://www.acba.am. Unlike Ameria, ACBA serves all content in the
initial HTML response (including tab content), so no Selenium is needed.
 
Data sources:
    - Loans:    https://www.acba.am/hy/individuals/loans/
    - Deposits: https://www.acba.am/hy/individuals/save-and-invest/deposits/
    - Branches: https://www.acba.am/hy/about-bank/Branches-and-ATMs
 
Output: Data/raw/acba/{loans,deposits,branches}.json
"""

import time
import re
import requests
from bs4 import BeautifulSoup
from banks.base_scraper import BaseScraper

BASE_URL = "https://www.acba.am"
HEADERS = {"User-Agent": "Mozilla/5.0"}


class ACBAScraper(BaseScraper):
    '''Scraper for ACBA Bank.'''
    def __init__(self):
        super().__init__("acba")

    def _get_soup(self, url):
        """Fetch a URL and return a BeautifulSoup object."""
        response = requests.get(url, headers=HEADERS, timeout=15)
        return BeautifulSoup(response.text, "html.parser")

    def _discover_links(self, start_url):
        """
        Load a page and return all links that start with the same URL prefix.
        e.g. start_url="https://www.acba.am/hy/individuals/loans/"
        returns all hrefs starting with that exact prefix (but not the page itself).
        """
        soup = self._get_soup(start_url)
        links = set()

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            
            # Skip anchors and javascript (links)
            if href.startswith("#") or href.startswith("javascript"):
                continue

            # Build full URL
            if href.startswith("http"):
                full = href
            elif href.startswith("/"):
                full = BASE_URL + href
            else:
                full = BASE_URL + "/" + href

            # Normalize domain variations
            full = full.replace("http://acba.am", "https://www.acba.am")
            full = full.replace("http://www.acba.am", "https://www.acba.am")
            full = full.replace("https://acba.am", "https://www.acba.am")

            # Only keep links that start with the same prefix
            if not full.startswith(start_url):
                continue

            # Skip if it's the listing page itself
            if full.rstrip("/") == start_url.rstrip("/"):
                continue

            # Skip files and tools
            if any(skip in full.lower() for skip in [
                ".pdf", ".xlsx", "calculator", "hashvich",
                "wizard.php", "acbadigital",
            ]):
                continue

            links.add(full)

        return sorted(links)

    def _extract_page(self, url):
        """
        Extracts product info from a page:
        - meta description
        - main description text
        - tab titles + tab content
        - tables
        """
        soup = self._get_soup(url)

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Meta description
        meta = soup.find("meta", {"name": "description"})
        meta_desc = meta["content"].strip() if meta and meta.get("content") else ""

        parts = [meta_desc]

        # Business card info
        for card in soup.find_all("div", class_="product__bus_cart__item-c"):
            card_title = card.find("div", class_="product__bus_cart__item-c__title")
            card_value = card.find("div", class_="product__bus_cart__item-c__sub_title")
            if card_title and card_value:
                parts.append(f"{card_title.get_text(strip=True)}: {card_value.get_text(strip=True)}")

        # Main description (before tabs)
        for div in soup.find_all("div", class_="txt__tpl1"):
            # Only grab ones that aren't inside tab bodies
            parent = div.find_parent(class_="tabs__tpl1__bodys__item")
            if not parent:
                text = div.get_text(separator="\n", strip=True)
                if text and len(text) > 20:
                    parts.append(text)

        # Tab content - pair tab titles with tab bodies
        tab_containers = soup.find_all("div", class_="tabs__tpl1")
        for container in tab_containers:
            tab_titles = container.find_all("div", class_="tabs__tpl1__tabs__item")
            tab_bodies = container.find_all("div", class_="tabs__tpl1__bodys__item")

            for i, (ttitle, tbody) in enumerate(zip(tab_titles, tab_bodies)):
                tab_name = ttitle.get_text(strip=True)
                tab_text = tbody.get_text(separator="\n", strip=True)
                if tab_text and len(tab_text) > 10:
                    parts.append(f"[{tab_name}]\n{tab_text}")

        # Also extract any tables not inside tabs
        for table in soup.find_all("table"):
            parent = table.find_parent(class_="tabs__tpl1__bodys__item")
            if parent:
                continue  # already captured in tab content
            rows = []
            for tr in table.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
                cells = [c for c in cells if c]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                parts.append("\n".join(rows))

        # Deduplicate lines and combine
        seen = set()
        clean_lines = []
        for part in parts:
            for line in part.split("\n"):
                line = line.strip()
                if line and line not in seen and len(line) > 3:
                    seen.add(line)
                    clean_lines.append(line)

        description = "\n".join(clean_lines)
        return title, description

    def _scrape_product_pages(self, links, product_type):
        """Scrape each product URL."""
        results = []
        for url in links:
            print(f"    {url}")
            try:
                title, description = self._extract_page(url)
                results.append({
                    "url": url,
                    "bank": "acba",
                    "type": product_type,
                    "name": title,
                    "description": description[:12000],
                })
            except Exception as e:
                print(f"    ERROR: {e}")
            time.sleep(1)
        return results

    def scrape_loans(self):
        print("Discovering loan links...")
        links = self._discover_links(f"{BASE_URL}/hy/individuals/loans/")
        print(f"Found {len(links)} loan pages")
        return self._scrape_product_pages(links, "credit")

    def scrape_deposits(self):
        print("Discovering deposit links...")
        links = self._discover_links(f"{BASE_URL}/hy/individuals/save-and-invest/deposits/")
        print(f"Found {len(links)} deposit pages")
        return self._scrape_product_pages(links, "deposit")

    def scrape_branches(self):
        print("Scraping branches...")
        url = f"{BASE_URL}/hy/about-bank/Branches-and-ATMs"
        soup = self._get_soup(url)
        results = []

        for branch in soup.find_all("div", class_="fb_branch"):
            name_el = branch.find("div", class_="fb_branch__head__title")
            city_el = branch.find("div", class_="fb_branch__place")

            # Address and hours are in fb_branch__list__item divs
            list_items = branch.find_all("div", class_="fb_branch__list__item")

            address = ""
            hours = ""
            cash_hours = ""
            for item in list_items:
                text = item.get_text(strip=True)
                if not text:
                    continue
                # First item with location icon is the address
                if item.find("img") and not address:
                    address = text
                elif "\u057d\u057a\u0561\u057d\u0561\u0580\u056f\u0578\u0582\u0574" in text.lower() or "\u0570\u0561\u0577\u057e" in text.lower():
                    cash_hours = text
                elif any(kw in text for kw in ["\u0535\u0580\u056f", "09:", "9:", "09\u0589", "9\u0589"]):
                    hours = text

            name = name_el.get_text(strip=True) if name_el else ""
            city = city_el.get_text(strip=True) if city_el else ""

            if not name:
                continue

            results.append({
                "url": url,
                "bank": "acba",
                "type": "branch",
                "name": name,
                "city": city,
                "address": address,
                "hours": hours,
                "cash_hours": cash_hours,
            })

        print(f"Found {len(results)} branches")
        return results