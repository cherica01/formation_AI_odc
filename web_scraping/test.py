import requests
from bs4 import BeautifulSoup

url = "https://www.lequipe.fr/Football/ligue-des-champions/page-calendrier-resultats"
soup = BeautifulSoup(requests.get(url).text, 'html.parser')

teams = [t.text.strip() for t in soup.find_all('div', class_='TeamScore__name')]
hours = soup.find_all('div', class_='TeamScore__data')

for i, hour in enumerate(hours):
    date = hour.find_previous('div', class_='caption caption--small').text.strip()
    eq1, eq2 = teams[2*i], teams[2*i + 1]
    print(f"{date} - {eq1} vs {eq2} à {hour.text.strip()}")