# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:19:50 2020

@author: krish
"""


"""
Evaluate the dataset of the Summer Olympics, London 2012 to:

Find and print the name of the country that won maximum gold medals,
Find and print the countries who won more than 20 gold medals,
Print the medal tally,
Print each country name with the corresponding number of gold medals, and
Print each country's name with the total number of medals won.
Click here to download the additional resources -  
"""

import pandas as pd
import numpy as np

olympic_data  = pd.read_excel('D:\PythonProject\SummerOlympics\Olympic 2012 Medal Tally.xlsx')

print(olympic_data)

gold_medals = np.array(olympic_data['Gold'])
bronze_medals = np.array(olympic_data['Bronze'])
silver_medals = np.array(olympic_data['Silver'])
countries = np.array(olympic_data['Country'])

print(countries)
print(gold_medals.argmax())
max_Gold = gold_medals.argmax()

country_with_max_gold = countries[max_Gold]
print(country_with_max_gold)

countries_won_gold_medals_above20 = countries[gold_medals > 20]

print(countries_won_gold_medals_above20)

print(olympic_data.columns[0],olympic_data.columns[3], olympic_data.columns[4], olympic_data.columns[5], 'Total')

for i in range(len(countries)):
    total_medals = gold_medals[i] + silver_medals[i] + bronze_medals[i]
    
    print(countries[i] , gold_medals[i] , silver_medals[i], bronze_medals[i], total_medals)
    

    