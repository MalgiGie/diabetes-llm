from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup

def google_search(query, api_key, cse_id, num_results=10):
    service = build("customsearch", "v1", developerKey=api_key)
    result = service.cse().list(q=query, cx=cse_id, num=num_results).execute()

    links = [item['link'] for item in result.get('items', [])]

    return links

def scrape_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        paragraphs = soup.find_all('p')
        text = "\n".join([p.get_text() for p in paragraphs])

        return text
    except Exception as e:
        print(f"Nie udało się pobrać strony: {url} - {str(e)}")
        return None

def clean_data(text):
    clean_text = text.replace("\n", " ").strip()
    return clean_text

def split_text(text, chunk_size=512):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

def get_data(query):
    with open('CUSTOM_SEARCH_API_KEY.txt', 'r', encoding='utf-8') as plik:
        API_KEY =  plik.read()
    with open('CSE_ID.txt', 'r', encoding='utf-8') as plik:
        CSE_ID =  plik.read()

    links = google_search(query, API_KEY, CSE_ID)

    processed_texts = []

    for index, link in enumerate(links, start=1):

        content = scrape_page(link)

        if content:
            cleaned_content = clean_data(content)

            for chunk in split_text(cleaned_content, chunk_size=512):
                processed_texts.append(chunk)

    return processed_texts



if __name__ == "__main__":
    with open('CUSTOM_SEARCH_API_KEY.txt', 'r', encoding='utf-8') as plik:
        API_KEY =  plik.read()
    with open('CSE_ID.txt', 'r', encoding='utf-8') as plik:
        CSE_ID =  plik.read()
    query = "Diabetes diet"

    links = google_search(query, API_KEY, CSE_ID)

    for index, link in enumerate(links, start=1):
        print(f"Scrapuję stronę {index}: {link}")
        content = scrape_page(link)
        if content:
            print(f"Zawartość strony {index}:\n{content[:500]}...\n")
