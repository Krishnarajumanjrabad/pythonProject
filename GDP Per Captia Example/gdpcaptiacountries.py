# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:39:55 2020

@author: krish
"""
"""
Problem Statement: Evaluate the dataset containing the GDPs of different countries to:

Find and print the name of the country with the highest GDP
Find and print the name of the country with the lowest GDP
Print out text and input values iteratively
Print out the entire list of the countries with their GDPs
Print the highest GDP value, lowest GDP value, mean GDP value, standardized GDP value, and the sum of all the GDPs
Follow the cues provided to complete the assignment.
"""



import numpy as np
countries = np.array(['Algeria','Angola','Argentina','Australia','Austria','Bahamas','Bangladesh','Belarus','Belgium','Bhutan','Brazil','Bulgaria','Cambodia','Cameroon','Chile','China','Colombia','Cyprus','Denmark','El Salvador','Estonia','Ethiopia','Fiji','Finland','France','Georgia','Ghana','Grenada','Guinea','Haiti','Honduras','Hungary','India','Indonesia','Ireland','Italy','Japan','Kenya', 'South Korea','Liberia','Malaysia','Mexico', 'Morocco','Nepal','New Zealand','Norway','Pakistan', 'Peru','Qatar','Russia','Singapore','South Africa','Spain','Sweden','Switzerland','Thailand', 'United Arab Emirates','United Kingdom','United States','Uruguay','Venezuela','Vietnam','Zimbabwe'
])
print(countries)

countries_per_captia = np.array([
2255.225482,629.9553062,11601.63022,25306.82494,27266.40335,19466.99052,588.3691778,2890.345675,24733.62696,1445.760002,4803.398244,2618.876037,590.4521124,665.7982328,7122.938458,2639.54156,3362.4656,15378.16704,30860.12808,2579.115607,6525.541272,229.6769525,2242.689259,27570.4852,23016.84778,1334.646773,402.6953275,6047.200797,394.1156638,385.5793827,1414.072488,5745.981529,837.7464011,1206.991065,27715.52837,18937.24998,39578.07441,478.2194906,16684.21278,279.2204061,5345.213415,6288.25324,1908.304416,274.8728621,14646.42094,40034.85063,672.1547506,3359.517402,36152.66676,3054.727742,33529.83052,3825.093781,15428.32098,33630.24604,39170.41371,2699.123242,21058.43643,28272.40661,37691.02733,9581.05659,5671.912202,757.4009286,347.7456605])

print(countries_per_captia)

# finding the max GDP from countries per captia income
max_gdp_per_captia = countries_per_captia.argmax()

# findin the correspondig country with max per captia income
country_max_per_captica = countries[max_gdp_per_captia]


#Printing the coutry with highest GDP
print(country_max_per_captica , countries_per_captia[max_gdp_per_captia])

#Printign the country with lowest GDP
min_gdp_per_captia = countries_per_captia.argmin()
print(countries[min_gdp_per_captia] , countries_per_captia[min_gdp_per_captia])

#Print out the text and input values iteratively
for i in countries:
    print(i)
    
for i in range(len(countries)):
    country = countries[i]
    country_with_gdp = countries_per_captia[i]
    
    print('evulating he country {} with GDP {}'.format(country, country_with_gdp))
    
# Check for max GDP
print(countries_per_captia.max())

#Check for Min GDP
print(countries_per_captia.min())

#Check for mean GDP
print(countries_per_captia.mean())

#Check for sum of all GDP
print(countries_per_captia.std())

print(countries_per_captia.sum())
