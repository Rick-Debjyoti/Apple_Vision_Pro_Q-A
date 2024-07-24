import requests
from bs4 import BeautifulSoup

def scrape_apple_vision_pro_landing_page(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page {url}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ""
    for paragraph in soup.find_all('p'):
        text += paragraph.get_text() + "\n"
    
    return text

if __name__ == "__main__":
    url = "https://www.apple.com/apple-vision-pro/" 
    text = scrape_apple_vision_pro_landing_page(url)
    with open("web_scraped_text.txt", "w") as file:
        file.write(text)
