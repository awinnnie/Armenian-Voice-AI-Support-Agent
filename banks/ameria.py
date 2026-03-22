"""
Ameria Bank scraper.
Scrapes loan products, deposit products and branch information from
https://ameriabank.am. Ameria's site loads product content dynamically
via JavaScript (WebSitesCreative CMS modules), so Selenium is required
for product pages. The nav menu HTML is available on page load.
 
Data sources:
    - Loans:    https://ameriabank.am/personal/loans
    - Deposits: https://ameriabank.am/personal/saving
    - Branches: https://ameriabank.am/service-network
 
Output: Data/raw/ameria/{loans,deposits,branches}.json
"""

import time
import json
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from banks.base_scraper import BaseScraper

BASE_URL = "https://ameriabank.am"


class AmeriaScraper(BaseScraper):
    '''Scraper for Ameria Bank'''
    
    def __init__(self):
        super().__init__("ameria")
        self.driver = None

    def _get_driver(self):
        """
        Get or create a headless Chrome driver.
        Created once, reused for all pages to avoid repeated startup cost.
        """
        if self.driver is None:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--lang=hy")
            options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            self.driver = webdriver.Chrome(options=options)
        return self.driver

    def _discover_nav_links(self, page_url):
        """
        Load a page with Selenium and extract product links from the nav menu.
 
        Ameria's nav uses a mega-menu with nested dropdowns. The currently
        active section (loans or saving) is marked with CSS class "current active".
        We extract links only from that section to avoid picking up cards,
        accounts, etc.
 
        Filtering rules:
            - Must be on ameriabank.am domain
            - Must have at least 3 URL path segments (product page, not category)
            - Skips: calculators, external sites (mycar.am, myhome.am),
              login pages, support pages, "see all" pages
        """
        
        driver = self._get_driver()
        driver.get(page_url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find the <li> that is "current active" - this is the nav section we're on
        active_section = soup.find("li", class_=re.compile(r"current.*active|active.*current"))
        if not active_section:
            # fallback: search all nav links
            active_section = soup

        links = set()
        for a in active_section.find_all("a", href=True):
            href = a["href"]
            # Skip anchors, external, calculators, support, login
            if href.startswith("#"):
                continue
            if not href.startswith("http"):
                href = BASE_URL + href
            if "ameriabank.am" not in href:
                continue
            # Skip non-product pages
            skip_words = [
                "hashvich", "calculator", "see-all", "leaflet",
                "support", "customer.", "join.", "payments.",
                "apply-now", "mycar.am", "myhome.am", "signin",
            ]
            if any(w in href.lower() for w in skip_words):
                continue
            # Must be a product page (has at least 4 path segments like /personal/loans/x/y)
            path = href.replace(BASE_URL, "")
            segments = [s for s in path.split("/") if s]
            if len(segments) >= 3:
                links.add(href)

        return sorted(links)

    def _extract_page_content(self, url, wait=5):
        """
        Loads a product page with Selenium and extract all visible content
 
        1. Load page and wait for JS modules to render
        2. Click all tab buttons to reveal hidden tab content
           (Ameria uses Bootstrap-style tabs with data-toggle="tab")
        3. Remove noise: scripts, styles, nav, header, footer, mega-menu
        4. Extract text from CMS containers:
           - wsc_content_manager_module_container (Ameria's CMS)
           - tab-content / tab-pane (Bootstrap tabs)
           - DNNModuleContent (DNN CMS wrapper)
        5. Extract tables as pipe-separated rows
        6. Deduplicate lines and combine
 
        Returns (title, description)
        """
        driver = self._get_driver()
        driver.get(url)
        time.sleep(wait)

        # Click all tab buttons to reveal hidden tab content
        for selector in [
            "a[data-toggle='tab']", ".nav-tabs a", ".nav-tabs li a",
            "a[role='tab']", ".wsc-tab-item",
        ]:
            try:
                tabs = driver.find_elements(By.CSS_SELECTOR, selector)
                for tab in tabs:
                    try:
                        driver.execute_script("arguments[0].click();", tab)
                        time.sleep(0.5)
                    except Exception:
                        pass
            except Exception:
                pass

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements
        for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
            tag.decompose()

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True).split("|")[0].strip() if title_tag else ""

        # Meta description
        meta = soup.find("meta", {"name": "description"})
        meta_desc = meta["content"].strip() if meta and meta.get("content") else ""

        # Remove nav/header/footer/menus to isolate main content
        for tag in soup.find_all(["nav", "header", "footer"]):
            tag.decompose()
        for tag in soup.find_all("div", id=re.compile(r"mainMenu")):
            tag.decompose()
        for tag in soup.find_all("ul", class_="dropdown-menu"):
            tag.decompose()

        # Extract text from Ameria's CMS content containers
        content_parts = []
        for selector in [
            "div.wsc_content_manager_module_container",
            "div.tab-content", "div.tab-pane",
            "div.c_contentpane", "div.DNNModuleContent",
        ]:
            for el in soup.select(selector):
                text = el.get_text(separator="\n", strip=True)
                if len(text) > 30:
                    content_parts.append(text)

        # Extract tables (terms, rates, conditions)
        table_texts = []
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
                cells = [c for c in cells if c]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                table_texts.append("\n".join(rows))

        # Combine, deduplicate lines
        all_text = meta_desc + "\n"
        seen = set()
        for part in content_parts + table_texts:
            for line in part.split("\n"):
                line = line.strip()
                if line and line not in seen and len(line) > 5:
                    seen.add(line)
                    all_text += line + "\n"

        return title, all_text.strip()

    def _scrape_product_pages(self, links, product_type):
        """
        Scrapes a list of product page URLs with Selenium.
 
        links: List of URLs to scrape
        product_type: "credit" or "deposit" (stored in JSON output)
 
        Returns list of dicts with keys: url, bank, type, name, description
        """
        results = []
        for url in links:
            print(f"    {url}")
            try:
                title, description = self._extract_page_content(url)
                results.append({
                    "url": url,
                    "bank": "ameria",
                    "type": product_type,
                    "name": title,
                    "description": description[:12000],
                })
            except Exception as e:
                print(f"    ERROR: {e}")
            time.sleep(1)
        return results

    def scrape_loans(self):
        print("Discovering loan links from nav...")
        links = self._discover_nav_links(f"{BASE_URL}/personal/loans")
        # Filter: keep only links under /loans/ or /loan/ path
        links = [l for l in links if "/loan" in l.replace(BASE_URL, "")]
        print(f"Found {len(links)} loan pages")
        return self._scrape_product_pages(links, "credit")

    def scrape_deposits(self):
        print("Discovering deposit links from nav...")
        links = self._discover_nav_links(f"{BASE_URL}/personal/saving")
        # Filter: keep only links under /saving/ or /savings/ or /accounts/.*saving
        links = [l for l in links if "/saving" in l.replace(BASE_URL, "")]
        print(f"Found {len(links)} deposit pages")
        return self._scrape_product_pages(links, "deposit")

    def scrape_branches(self):
        """
        Scrapes all branches by cycling through each region in the dropdown.
 
        The /service-network page has a Google Maps widget with:
        - A region <select> dropdown (Yerevan is default)
        - A "List" toggle (div#map-menu-toggle) that opens a sidebar
        - Branch cards in #locations-list with class .sidebar-item
 
        For each region:
        1. Select the region from the dropdown
        2. Wait 3s for the sidebar to refresh (JS filters markers + list)
        3. Extract only visible (.is_displayed()) branch cards
           (hidden cards belong to other regions)
 
        Branch card CSS selectors:
            .sidebar-item__title  -   branch name
            .sidebar-item__location - full address
            .sidebar-item__phone  -   phone number
            .sidebar-item__tag    -   hours type (standard/extended)
        """
        from selenium.webdriver.support.ui import Select
 
        print("  Loading service network page...")
        driver = self._get_driver()
        driver.get(f"{BASE_URL}/service-network")
        time.sleep(5)
 
        # Open the sidebar list once
        try:
            toggle = driver.find_element(By.ID, "map-menu-toggle")
            driver.execute_script("arguments[0].click();", toggle)
            time.sleep(2)
        except Exception:
            pass
 
        results = []
 
        # Find the region dropdown
        region_select_el = None
        for sel in driver.find_elements(By.TAG_NAME, "select"):
            options = sel.find_elements(By.TAG_NAME, "option")
            texts = [o.text.strip() for o in options if o.text.strip()]
            if any("\u0535\u0580\u0587\u0561\u0576" in t for t in texts):
                region_select_el = sel
                break
 
        if not region_select_el:
            print("    Could not find region dropdown!")
            return []
 
        select_obj = Select(region_select_el)
        region_options = [(o.get_attribute("value"), o.text.strip()) for o in select_obj.options]
        region_options = [(v, t) for v, t in region_options if t and t != "---"]
 
        for value, region_name in region_options:
            print(f"    Region: {region_name}")
 
            # Select region
            select_obj.select_by_value(value)
            time.sleep(3)  # wait for sidebar to refresh
 
            # Extract branch cards from sidebar
            cards = driver.find_elements(By.CSS_SELECTOR, "#locations-list .sidebar-item")
            count = 0
            for card in cards:
                try:
                    # Check if the card is actually visible
                    if not card.is_displayed():
                        continue
 
                    name_el = card.find_elements(By.CSS_SELECTOR, ".sidebar-item__title")
                    addr_el = card.find_elements(By.CSS_SELECTOR, ".sidebar-item__location")
                    phone_el = card.find_elements(By.CSS_SELECTOR, ".sidebar-item__phone")
                    tag_el = card.find_elements(By.CSS_SELECTOR, ".sidebar-item__tag")
 
                    name = name_el[0].text.strip() if name_el else ""
                    address = addr_el[0].text.strip() if addr_el else ""
                    phone = phone_el[0].text.strip() if phone_el else ""
                    hours = tag_el[0].text.strip() if tag_el else ""
 
                    if not name or len(name) < 3:
                        continue
 
                    results.append({
                        "url": f"{BASE_URL}/service-network",
                        "bank": "ameria",
                        "type": "branch",
                        "name": name,
                        "address": address,
                        "phone": phone,
                        "hours": hours,
                        "region": region_name,
                    })
                    count += 1
                except Exception:
                    pass
 
            if count > 0:
                print(f"      Found {count} branches")
 
        # Deduplicate by name+address
        seen = set()
        unique = []
        for b in results:
            key = b["name"] + b["address"]
            if key not in seen:
                seen.add(key)
                unique.append(b)
 
        return unique
 

    
    def scrape_all(self):
        """
        Run all scrapers and ensure the Selenium driver is closed afterward.
        """
        try:
            return super().scrape_all()
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None