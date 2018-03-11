
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 21:07:25 2017

@author: 11S14052 (HTTP 14)
"""

from bs4 import BeautifulSoup
import requests

quote_page = 'https://us.soccerway.com/teams/spain/spain/2137/matches/'
response = requests.get(quote_page)

# page = urllib3.connection_from_url(quote_page)
soup = BeautifulSoup(response.text, "lxml")

right_table=soup.find('table', class_='matches')
A=[]
B=[]
C=[] 
D=[]
E=[]
F=[]

for row in right_table.findAll("tr"):
        # The rows we care about don't have the th tag
        if row.th:
            continue
        cells = row.find_all('td')
        A.append(cells[0].text.strip())
        B.append(cells[1].text.strip())
        C.append(cells[2].text.strip())
        D.append(cells[3].text.strip())
        E.append(cells[4].text.strip())
        F.append(cells[5].text.strip())
        


#import pandas to convert list to data frame
import pandas as pd
df=pd.DataFrame(A,columns=['Day'])
df['Date']=B
df['Compettion']=C
df['Team A']=D
df['Score']=E
df['Team B']=F
df
