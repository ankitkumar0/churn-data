# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:32:34 2019

@author: Deepak Gupta
"""
import webbrowser
import pandas as pd
import os
import requests
os.chdir("E:\\")
text=pd.read_csv("output.csv")
a=text["words"][0]
new=2
taburl="http://google.com/?#q="
result=webbrowser.open(taburl+a,new=new)
from bs4 import BeautifulSoup
page1 = requests.get("https://www.google.com/search?q=academic+reputation&oq=academic+reputation&aqs=chrome..69i57j69i59j69i60l2.252j0j7&sourceid=chrome&ie=UTF-8")
src=page1.content
soup= BeautifulSoup(src,"lxml")
links=soup.find_all("a")
for link in links:
    if "href" in  link.attrs:
        print(str(link.attrs["href"])+"\n")
        

b=text["words"][1]
new=2
taburl="http://google.com/?#q="
result=webbrowser.open(taburl+b,new=new)
page2=requests.get("https://www.google.com/search?q=Academic+Reputation+Tracker&oq=Academic+Reputation+Tracker&aqs=chrome..69i57.765j0j7&sourceid=chrome&ie=UTF-8")
src=page2.content
soup= BeautifulSoup(src,"lxml")
links=soup.find_all("a")
for link in links:
    if "href" in  link.attrs:
        print(str(link.attrs["href"])+"\n")

c=text["words"][2]
new=2
taburl="http://google.com/?#q="
result=webbrowser.open(taburl+c,new=new)        
page3=requests.get("https://www.google.com/search?q=analyze+academic+reputation&oq=analyze+academic+reputation&aqs=chrome..69i57.345j0j7&sourceid=chrome&ie=UTF-8")
src=page3.content
soup= BeautifulSoup(src,"lxml")
links=soup.find_all("a")
for link in links:
    if "href" in  link.attrs:
        print(str(link.attrs["href"])+"\n")
 
d=text["words"][3]
new=2
taburl="http://google.com/?#q="
result=webbrowser.open(taburl+d,new=new)       
page4=requests.get("https://www.google.com/search?q=benefits+of+academic+reputation&oq=benefits+of+academic+reputation&aqs=chrome..69i57.361j0j7&sourceid=chrome&ie=UTF-8")
src=page4.content
soup= BeautifulSoup(src,"lxml")
links=soup.find_all("a")
for link in links:
    if "href" in  link.attrs:
        print(str(link.attrs["href"])+"\n")  

e=text["words"][4]
new=2
taburl="http://google.com/?#q="
result=webbrowser.open(taburl+e,new=new)        
page5=requests.get("https://www.google.com/search?q=how+to+improve+academic+reputation&oq=how+to+improve+academic+reputation&aqs=chrome..69i57j69i60.272j0j7&sourceid=chrome&ie=UTF-8")
src=page5.content
soup= BeautifulSoup(src,"lxml")
links=soup.find_all("a")
for link in links:
    if "href" in  link.attrs:
        print(str(link.attrs["href"])+"\n")      

f=text["words"][5]
new=2
taburl="http://google.com/?#q="
result=webbrowser.open(taburl+f,new=new)        
page6=requests.get("https://www.google.com/search?q=Employer+Reputation+Tracker&oq=Employer+Reputation+Tracker&aqs=chrome..69i57.1441j0j7&sourceid=chrome&ie=UTF-8")
src=page6.content
soup= BeautifulSoup(src,"lxml")
links=soup.find_all("a")
for link in links:
    if "href" in  link.attrs:
        print(str(link.attrs["href"])+"\n") 
  
g=text["words"][6]
new=2
taburl="http://google.com/?#q="
result=webbrowser.open(taburl+g,new=new)      
page7=requests.get("https://www.google.com/search?q=Employer+Reputation&oq=Employer+Reputation&aqs=chrome..69i57j69i59.232j0j7&sourceid=chrome&ie=UTF-8")
src=page7.content
soup= BeautifulSoup(src,"lxml")
links=soup.find_all("a")
for link in links:
    if "href" in  link.attrs:
        print(str(link.attrs["href"])+"\n")

h=text["words"][7]
new=2
taburl="http://google.com/?#q="
result=webbrowser.open(taburl+h,new=new)
page8=requests.get("https://www.google.com/search?q=benefits+of+employer+reputation+tracker&oq=benefits+of+employer+reputation+tracker&aqs=chrome..69i57.252j0j7&sourceid=chrome&ie=UTF-8")
src=page8.content
soup= BeautifulSoup(src,"lxml")
links=soup.find_all("a")
for link in links:
    if "href" in  link.attrs:
        print(str(link.attrs["href"])+"\n")     

i=text["words"][8]
new=2
taburl="http://google.com/?#q="
result=webbrowser.open(taburl+i,new=new)        
page9=requests.get("https://www.google.com/search?q=subject+rankings+tracker&oq=subject+rankings+tracker&aqs=chrome..69i57.298j0j7&sourceid=chrome&ie=UTF-8")
src=page9.content
soup= BeautifulSoup(src,"lxml")
links=soup.find_all("a")
for link in links:
    if "href" in  link.attrs:
        print(str(link.attrs["href"])+"\n")  

j=text["words"][9]
new=2
taburl="http://google.com/?#q="
result=webbrowser.open(taburl+j,new=new)        
page10=requests.get("https://www.google.com/search?q=subject+rank+tracker+tool&oq=subject+rank+tracker+tool&aqs=chrome..69i57.317j0j7&sourceid=chrome&ie=UTF-8")
src=page10.content
soup= BeautifulSoup(src,"lxml")
links=soup.find_all("a")
for link in links:
    if "href" in  link.attrs:
        print(str(link.attrs["href"])+"\n")   


        
        
        




