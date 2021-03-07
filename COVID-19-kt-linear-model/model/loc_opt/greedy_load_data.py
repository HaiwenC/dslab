"""
This load data module is based on the data dictionary of SafeGraph's Weekly Pattern v2
https://docs.safegraph.com/docs/weekly-patterns
"""

import pandas as pd


def extract_city_data(city_name = 'Los Angeles'):
    '''
    Extract the data of a certain city from the complete dataset
    This function might be slow since the complete dataset is large. Only run it once to extract the city data.
    :param city_name: the name of the target city
    :return: None
    '''
    # This should be the URL of the complete dataset
    data = pd.read_csv("./data/2020-05-18-weekly-patterns.csv")
    df = pd.DataFrame(data)
    select_LA = df.loc[df['city'] == city_name]
    select_LA.to_csv('./data/city_data.csv', encoding='utf-8')

class DLoader:
    def __init__(self):
        self.data = pd.DataFrame(pd.read_csv("./data/city_data.csv"))
        self.mdata = self.data.to_numpy()
        self.loc_ids = self.mdata[:,1]
        self.loc_names = self.mdata[:,2]
        self.addresses = self.mdata[:,3]
        self.visitor_cnt = self.mdata[:13]



if __name__ == '__main__':
    # extract_city_data()
    data = pd.read_csv("./data/city_data.csv")
    print(data.to_numpy()[:,14])
    pass







