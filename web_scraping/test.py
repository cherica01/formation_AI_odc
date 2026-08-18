import requests
from bs4 import BeautifulSoup

# 1. Saisir l'URL
url = "https://cherica.vercel.app/"

# 2. Importer le code de la page
response = requests.get(url)

# 3. Parser le HTML
soup = BeautifulSoup(response.text, "html.parser")

# 4. Rechercher tous les éléments ciblés (ex: tous les liens ou blocs de la page)
items = soup.find_all("a")  # On cherche toutes les balises <a> (liens)

# 5. Créer la liste vide
news_items = []

# 6. Parcourir chaque élément trouvé
for i in items:
    news_i = {}
    
    # Extraire le texte du lien s'il existe, sinon mettre "Sans titre"
    news_i['title'] = i.text.strip() if i.text else "Sans titre"
    
    # Extraire l'adresse de destination (attribut href)
    news_i['link'] = i.get('href', 'Pas de lien')
    
    # Ajouter le dictionnaire dans la liste
    news_items.append(news_i)

# 7. Affichage du résultat structuré
print(news_items)