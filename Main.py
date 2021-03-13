import requests
from bs4 import BeautifulSoup
import Datahandler



query = 'atrial fibrillation'

url = 'https://scholar.google.com/scholar?q=' + query + '&ie=UTF-8&oe=UTF-8&hl=en&btnG=Search'

content = requests.get(url).text
page = BeautifulSoup(content, 'lxml')
results = []
for entry in page.find_all("div", attrs={"class": "gs_ri"}): #tag containing both h3 and citation
    results.append({"title": entry.h3.a.text, "url": entry.a['href'], "citation": entry.find("div", attrs={"class": "gs_rs"}).text})

print(f'{results}')