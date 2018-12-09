import urllib.request
import urllib.response
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import time


def getimg(prod):
    try:
        return prod.find('img')['src']
    except:
        return None


browser = webdriver.Chrome()

browser.get("https://6dollarshirts.com/guys-tees?page=4")
time.sleep(1)

# lenOfPage = browser.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
# match=False
# while(match==False):
#     lastCount = lenOfPage
#     time.sleep(3)
#     lenOfPage = browser.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
#     if lastCount==lenOfPage:
#         match=True

elem = browser.find_element_by_tag_name("body")

no_of_pagedowns = 120

while no_of_pagedowns:
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5)
    no_of_pagedowns-=1


source_data = browser.page_source

soup = BeautifulSoup(source_data)

# result_box = soup.find('ul', attrs={'id': 'search-result-items'})
prods = soup.find_all('div', attrs={'class': 'image'})
print(len(prods))
print(prods[0].find('img')['src'])
images=list(map(lambda x: getimg(x),prods))

images = [x for x in images if x is not None]

commands=list(map(lambda x: "wget https://6dollarshirts.com/"+x,images))

print(commands[0])

os.chdir('./images/men')
for com in commands:
    os.system(com)
    time.sleep(0.2)