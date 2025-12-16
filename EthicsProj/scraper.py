# scraper.py
import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        article_text = "\n".join(paragraphs)
        return article_text[:4000]
    except Exception as e:
        print(f"[!] Error fetching article: {e}")
        return None
