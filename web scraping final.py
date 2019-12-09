# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 21:04:35 2019

@author: Deepak Gupta
"""

from selenium import webdriver
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq
myurl="https://www.flipkart.com/search?q=i%20phone&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"
uclient=uReq(myurl)
page_html=uclient.read()
uclient.close()
page_soup=soup(page_html,"html.parser")
container=page_soup.find_all("div",{"class":"_3O0U0u"})
containers=container[0]
print(len(container))
print(soup.prettify(container[0]))
print(containers.div.img["alt"])
price=containers.find_all("div",{"class":"col col-5-12 _2o7WAb"})
print(price[0].text)
rating=containers.find_all("div",{"class":"niH0FQ"})
print(rating[0].text)
filename="products.csv"
f=open(filename,"w")
header="product_name,price,rating\n"
f.write(header)
for containers in container:
    product_name=containers.div.img["alt"]
    price_container=containers.find_all("div",{"class":"col col-5-12 _2o7WAb"})
    price=price_container[0].text.strip()
    rating_container=containers.find_all("div",{"class":"niH0FQ"})
    rating=rating_container[0].text
    
    
print("product_name:" + product_name)    
print("price"+price)
print("rating: "+rating)
#string parsing
trim_price="".join(price.split(","))
rm_rupee=trim_price.split("₹")
add_rm_price="rs"+rm_rupee[1]
split_price=add_rm_price.split("E")
final_prices=split_price[0]
split_rating=rating.split(" ")
final_rating=split_rating[0]
print(product_name.replace(",","|") + final_prices + ","+ final_rating +"\n" )




    