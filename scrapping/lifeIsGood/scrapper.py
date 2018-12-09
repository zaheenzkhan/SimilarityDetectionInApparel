import urllib.request
import urllib.response
from bs4 import BeautifulSoup
import os
import time


page="https://www.lifeisgood.com/men/graphic-tees/"


page = urllib.request.urlopen(page)
soup = BeautifulSoup(page, 'html.parser')

result_box = soup.find('ul', attrs={'id': 'search-result-items'})
prods = soup.find_all('div', attrs={'class': 'product-image'})
print(len(prods))

images=list(map(lambda x: x.find('img')['data-src'],prods))

commands=list(map(lambda x: "wget "+x,images))

print(commands[0])

os.chdir('./images/women')
for com in commands:
    os.system(com)
    time.sleep(0.3)