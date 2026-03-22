from banks.evoca import EvocaScraper
from banks.ameria import AmeriaScraper  
from banks.acba import ACBAScraper   

scrapers = [
    EvocaScraper(),
    AmeriaScraper(),
    ACBAScraper(),
]

for scraper in scrapers:
    scraper.scrape_all()
    
