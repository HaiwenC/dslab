import pandas as pd

population = pd.read_csv('city_community_table-2.csv').iloc[:,[1,2,11]]
#print(df.geo_merge.unique())
population['geo_merge'] = population['geo_merge'].str.replace('City of ', '')
population['geo_merge'] = population['geo_merge'].str.replace('Los Angeles -', '')
#print(df.geo_merge.unique())
print(population)