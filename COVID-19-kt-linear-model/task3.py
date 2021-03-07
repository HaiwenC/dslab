import pandas as pd
from IPython.display import display
from IPython.display import Image

df = pd.read_csv('latimes-place-totals.csv')
#print(df)

lacountydata = df.loc[df['county'] == 'Los Angeles']
print(lacountydata)

regions = lacountydata.place.unique()
print(regions)
print(len(regions))

print(lacountydata.groupby('place').first())
# lacountydata.to_csv('lacountydataGroupbyPlace.csv')
print(lacountydata.iloc[:,[0,3,4,6,7]])


# #task 2
# population = pd.read_csv('city_community_table-2.csv').iloc[:,[1,2,11]]
population = pd.read_csv('LAtimesNeighborhoodPop.csv')
#print(df.geo_merge.unique())
population['neighborhood'] = population['neighborhood'].str.lower() 
population['neighborhood'] = population['neighborhood'].str.replace('-', ' ')
population=population[population.population != -1]
# population['geo_merge'] = population['geo_merge'].str.replace('City of ', '')
# population['geo_merge'] = population['geo_merge'].str.replace('Los Angeles -', '')
#print(df.geo_merge.unique())
print(population)

popset = set(population.neighborhood)
lacountydata = lacountydata.rename(columns={'x': 'Longitude', 'y': 'Latitude'})
lacountydata['place'] = lacountydata['place'].str.lower() 
lacountydata['place'] = lacountydata['place'].str.replace('-', ' ')
lacountyset = set(lacountydata.place)
print(len(popset))
print(len(lacountyset))

print("set1 Intersect set2 size: ", len(popset.intersection(lacountyset)))
print("set1 Intersect set2 : ", popset.intersection(lacountyset))
print ("set1 has, but set2 dont have: ",popset.difference(lacountyset)) 
print ("set2 has, but set1 dont have: ",lacountyset.difference(popset)) 

comb = pd.merge(population, lacountydata, left_on='neighborhood', right_on='place',how='inner')
print(comb)
# print(len(set(comb.place)))
print(comb.iloc[:,[2,0,6,1,8,9]])
comb.iloc[:,[2,0,6,1,8,9]].to_csv('task2_3.csv')

print(comb.groupby('neighborhood').first().iloc[:,[0]])
comb.groupby('neighborhood').first().iloc[:,[0]].to_csv('task2.csv')

location = comb.groupby('neighborhood').first().iloc[:,[8,7]]
print(location)
location.to_csv('neighborhoodLocation.csv')