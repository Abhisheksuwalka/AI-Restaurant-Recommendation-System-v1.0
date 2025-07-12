import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import random

class RestaurantScraper:
    """Scrapes restaurant data from a public website (e.g., Yelp, Zomato, OpenTable)"""
    def __init__(self, base_url, max_pages=5, delay=2):
        self.base_url = base_url
        self.max_pages = max_pages
        self.delay = delay
        self.restaurants = []
        self.reviews = []
        self.ratings = []

    def scrape(self):
        print(f"Starting scrape from {self.base_url}...")
        for page in range(1, self.max_pages + 1):
            url = f"{self.base_url}?page={page}"
            print(f"Scraping: {url}")
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to fetch {url}")
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            self._parse_restaurants(soup)
            time.sleep(self.delay)
        print(f"Scraping complete. Found {len(self.restaurants)} restaurants.")

    def _parse_restaurants(self, soup):
        # Example for Yelp-like structure. Adjust selectors for your target site.
        for card in soup.select('.restaurant-card'):
            name = card.select_one('.restaurant-name').get_text(strip=True)
            cuisine = card.select_one('.cuisine-type').get_text(strip=True)
            location = card.select_one('.location').get_text(strip=True)
            price_range = card.select_one('.price-range').get_text(strip=True)
            rating = float(card.select_one('.rating').get_text(strip=True))
            num_reviews = int(card.select_one('.review-count').get_text(strip=True))
            restaurant_id = len(self.restaurants) + 1
            self.restaurants.append({
                'restaurant_id': restaurant_id,
                'name': name,
                'cuisine': cuisine,
                'location': location,
                'price_range': price_range,
                'rating': rating,
                'num_reviews': num_reviews
            })
            # Simulate reviews and ratings
            for i in range(random.randint(5, 20)):
                review_text = f"Sample review {i+1} for {name}"
                review_rating = round(random.uniform(1, 5), 1)
                user_id = random.randint(1, 1000)
                self.reviews.append({
                    'user_id': user_id,
                    'restaurant_id': restaurant_id,
                    'review_text': review_text,
                    'rating': review_rating,
                    'timestamp': pd.Timestamp.now()
                })
                self.ratings.append({
                    'user_id': user_id,
                    'restaurant_id': restaurant_id,
                    'rating': review_rating,
                    'timestamp': pd.Timestamp.now()
                })

    def save_to_csv(self, data_dir='data'):
        pd.DataFrame(self.restaurants).to_csv(f'{data_dir}/restaurants.csv', index=False)
        pd.DataFrame(self.reviews).to_csv(f'{data_dir}/reviews.csv', index=False)
        pd.DataFrame(self.ratings).to_csv(f'{data_dir}/ratings.csv', index=False)
        print(f"Saved scraped data to {data_dir}/")

if __name__ == '__main__':
    # Example usage: scrape first 5 pages from a Yelp-like site
    scraper = RestaurantScraper(base_url='https://example.com/restaurants', max_pages=5)
    scraper.scrape()
    scraper.save_to_csv()
