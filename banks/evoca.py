"""
Evoca Bank scraper.

Scrapes loan products, deposit products, and branch information from
https://www.evoca.am. Evoca serves all content in the initial HTML
response (no JavaScript rendering needed), so plain requests and 
BeautifulSoup is sufficient. No Selenium required.

Data sources:
    - Loans:    https://www.evoca.am/hy/loans/
                + https://www.evoca.am/hy/credit-history-and-score (info page)
                + https://www.evoca.am/hy/loans-important-information (info page)
    - Deposits: https://www.evoca.am/hy/deposits/
                + https://www.evoca.am/hy/deposits-important-information (info page)
    - Branches: https://www.evoca.am/hy/branches-and-atms?type=branches

Output: Data/raw/evoca/{loans,deposits,branches}.json
"""

import requests
import time
from bs4 import BeautifulSoup
from banks.base_scraper import BaseScraper

BASE_URL = "https://www.evoca.am"
HEADERS = {"User-Agent": "Mozilla/5.0"}

class EvocaScraper(BaseScraper):
    """Scraper for Evoca Bank."""
    def __init__(self):
        super().__init__("evoca")

    def _get_soup(self, url):
        """Fetch a URL and return a BeautifulSoup object."""
        response = requests.get(url, headers=HEADERS)
        return BeautifulSoup(response.text, "html.parser")

    def _get_all_links(self, section):
        """
        Get all product page links for a given section.

        Loads the section listing page (e.g., /hy/loans/) and finds
        all <a href> links that contain "/hy/{section}/" in the path.

        section: "loans" or "deposits"

        Returns list of full URLs like:
            ["https://www.evoca.am/hy/loans/car-loans/car-loans", ...]
        """
        soup = self._get_soup(f"{BASE_URL}/hy/{section}")
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if f"/hy/{section}/" in href:
                full = BASE_URL + href if href.startswith("/") else href
                links.add(full)
        return list(links)

    def _scrape_product_page(self, url, topic):
        """
        Extract product info from a single Evoca product page.

        url: Full URL of the product page
        topic: "credit" or "deposit" (stored in output JSON)

        Returns a dict with: url, bank, type, name, description,
        plus dynamic keys from the summary table
        """
        soup = self._get_soup(url)
        data = {"url": url, "bank": "evoca", "type": topic}

        # Product name from <h1>
        title = soup.find("h1")
        data["name"] = title.get_text(strip=True) if title else ""

        items = soup.find_all("tr", class_="cards-info__currency-item")
        for item in items:
            value = item.find("div", class_="cards-info__currency-price")
            label = item.find("div", class_="cards-info__currency-desc")
            if value and label:
                data[label.get_text(strip=True)] = value.get_text(strip=True)

        # Product description text
        desc = soup.find("div", class_="cards-info__desc")
        data["description"] = desc.get_text(strip=True) if desc else ""

        return data

    def scrape_loans(self):
        """
        Scrape all loan products plus supplementary info pages.
        """
        # Flat info pages
        extra_pages = [
            f"{BASE_URL}/hy/credit-history-and-score",
            f"{BASE_URL}/hy/loans-important-information"
        ]
        results = []
        for url in extra_pages:
            soup = self._get_soup(url)
            body = soup.find("main") or soup.find("body")
            results.append({
                "url": url,
                "bank": "evoca",
                "type": "credit",
                "name": url.split("/")[-1],
                "description": body.get_text(strip=True) if body else ""
            })

        # Individual loan pages
        links = self._get_all_links("loans")
        for link in links:
            print(f"    {link}")
            results.append(self._scrape_product_page(link, "credit"))
            time.sleep(1)
        return results

    def scrape_deposits(self):
        """
        Scrape all deposit products plus the deposits info page.
        """
        # Flat info page
        extra_pages = [
            f"{BASE_URL}/hy/deposits-important-information"
        ]
        results = []
        for url in extra_pages:
            soup = self._get_soup(url)
            body = soup.find("main") or soup.find("body")
            results.append({
                "url": url,
                "bank": "evoca",
                "type": "deposit",
                "name": url.split("/")[-1],
                "description": body.get_text(strip=True) if body else ""
            })

        # Individual deposit pages
        links = self._get_all_links("deposits")
        for link in links:
            print(f"    {link}")
            results.append(self._scrape_product_page(link, "deposit"))
            time.sleep(1)
        return results

    def scrape_branches(self):
        """
        Scrape all Evoca branch locations.

        Each branch is a div.map-box__info with data-groupby="bank-branches".

        Branch card structure:
            h3.map-box__inner-title              - branch name
            li.map-box__inner-list-item--location - address
            li.map-box__inner-list-item--tel      - phone
            li.map-box__inner-list-item--working-days - hours
            data-city attribute                   - region
            data-community attribute              - community/district
        """
        url = f"{BASE_URL}/hy/branches-and-atms?type=branches"
        soup = self._get_soup(url)
        results = []

        for branch in soup.find_all("div", class_="map-box__info", attrs={"data-groupby": "bank-branches"}):
            name_tag = branch.find("h3", class_="map-box__inner-title")
            hours_tag = branch.find("li", class_="map-box__inner-list-item--working-days")
            address_tag = branch.find("li", class_="map-box__inner-list-item--location")
            phone_tag = branch.find("li", class_="map-box__inner-list-item--tel")

            results.append({
                "url": url,
                "bank": "evoca",
                "type": "branch",
                "name": name_tag.get_text(strip=True) if name_tag else "",
                "hours": hours_tag.get_text(strip=True) if hours_tag else "",
                "address": address_tag.get_text(strip=True) if address_tag else "",
                "phone": phone_tag.get_text(strip=True) if phone_tag else "",
                "region": branch.get("data-city", ""),
                "community": branch.get("data-community", ""),
            })

        return results