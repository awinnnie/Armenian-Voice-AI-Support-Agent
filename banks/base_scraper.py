"""
Abstract base class for all bank scrapers.

Defines the interface that each bank scraper (Evoca, Ameria, ACBA) must
implement: scrape_loans(), scrape_deposits(), scrape_branches().

- Output directory creation (Data/raw/<bank_name>/)
- JSON serialization with Armenian character support
- scrape_all() that runs all three scrapers and saves results

Usage:
    class MyBankScraper(BaseScraper):
        def __init__(self):
            super().__init__("mybank")

        def scrape_loans(self): ...
        def scrape_deposits(self): ...
        def scrape_branches(self): ...

    MyBankScraper().scrape_all() 
"""

from abc import ABC, abstractmethod
import json
import os

class BaseScraper(ABC):
    
    def __init__(self, bank_name):
        self.bank_name = bank_name
        self.output_dir = f"Data/raw/{bank_name}"
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def scrape_loans(self) -> list[dict]:
        """Scrape all loan products. Must be implemented by each bank scraper."""
        pass

    @abstractmethod
    def scrape_deposits(self) -> list[dict]:
        """Scrape all deposit products. Must be implemented by each bank scraper."""
        pass

    @abstractmethod
    def scrape_branches(self) -> list[dict]:
        """Scrape all branch locations. Must be implemented by each bank scraper."""
        pass

    def scrape_all(self):
        """
        Run all three scrapers and save results to JSON files.
        Called by scraper.py for each bank.

        Returns dict with keys: loans, deposits, branches
        """
        print(f"\n{'='*40}")
        print(f"Scraping {self.bank_name}...")
        
        print("Loans...")
        loans = self.scrape_loans()
        self._save(loans, "loans")

        print("Deposits...")
        deposits = self.scrape_deposits()
        self._save(deposits, "deposits")

        print("Branches...")
        branches = self.scrape_branches()
        self._save(branches, "branches")

        print(f"Done. {len(loans)} loans, {len(deposits)} deposits, {len(branches)} branches")
        return {"loans": loans, "deposits": deposits, "branches": branches}

    def _save(self, data, topic):
        """
        Save scraped data to a JSON file.

        data: List of dicts to serialize
        topic: Filename stem — "loans", "deposits", or "branches"

        Uses ensure_ascii=False to preserve Armenian characters in output.
        """
        path = f"{self.output_dir}/{topic}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)