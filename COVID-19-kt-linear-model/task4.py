import os
import pandas as pd

#read all files in directory
path = './'
files = []
data = {}
# r=root, d=directories, f = files
count = 1
for r, d, f in os.walk(path):
    for file in f:
        if '.gz' in file:
            filepath = os.path.join(r, file)
            files.append(filepath)
            print(file[:10],count)
            data[file[:10]]= pd.read_csv(filepath, compression='gzip',error_bad_lines=False).loc[:, ['origin_census_block_group', 'date_range_start', 'date_range_end','destination_cbgs']]
            count +=1
        
files.sort()
for f in files:
    print(f)
print(len(files))
print(data)


#data = pd.read_csv(files[0], compression='gzip',error_bad_lines=False).loc[:, ['origin_census_block_group', 'date_range_start', 'date_range_end','destination_cbgs']]
#print(type(data))
#d = {'55-1':data}
#print(d)
#print(type(d['55-1']))

