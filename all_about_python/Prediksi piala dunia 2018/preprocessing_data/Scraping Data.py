# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 09:56:38 2018

@author: Wahyu Nainggolan
"""

from bs4 import BeautifulSoup
import requests

quote_page = 'http://www.pesmaster.com/switzerland/pes-2016/team/11/'
response = requests.get(quote_page)

# page = urllib3.connection_from_url(quote_page)
soup = BeautifulSoup(response.text, "lxml")

right_table=soup.find('div', class_='squad-table-container')
A=[]
B=[]
C=[] 
D=[]
E=[]
F=[]
G=[] 
H=[] 

for row in right_table.findAll("tr"):
        # The rows we care about don't have the th tag
        if row.th:
            continue
        cells = row.find_all('td')
        G.append(cells[3].text.strip())
        A.append(cells[7].text.strip())
        B.append(cells[8].text.strip())
        C.append(cells[9].text.strip())
        D.append(cells[10].text.strip())
        E.append(cells[11].text.strip())
        F.append(cells[12].text.strip())
        H.append(cells[0].text.strip())

#import pandas to convert list to data frame
import pandas as pd
df=pd.DataFrame(A,columns=['Passing'])
df['Shoot']=B
df['Physic']=C
df['Defence']=D
df['Speed']=E
df['Dribbling']=F
df['Age']=G
df['Name']=H

df
data=df.iloc[1:]
data=data.to_csv("C:/Users/Wahyu Nainggolan/Documents/TA Gue/TA_2/Implementasi/Data/2018/data_train/country/swiss.csv")