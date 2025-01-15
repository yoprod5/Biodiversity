import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from sklearn.metrics.pairwise import pairwise_distances
data_path = '/Users/yoprod/Desktop/MesRecherches/GeoClef/'

validation_path = data_path+'Presence_Absence_surveys/Presences_Absences_train.csv'
# Load the data from the CSV file
#data['speciesId'] = data['speciesId'].astype(float)

data = pd.read_csv(validation_path, sep=";", header='infer', low_memory=True)
grouped_data = data.groupby(['lat', 'lon'])['speciesId'].apply(lambda x: ' '.join(map(str, x))).reset_index()
print(grouped_data.head)
df = grouped_data[['lon', 'lat', 'speciesId']]
df2 = pd.DataFrame(columns=['Id','Jaccard'])

Ind_jaccard = []
sample_id = []

vectors = [row.split() for row in grouped_data['speciesId']]
vectors = [[float(value) for value in row] for row in vectors]

print(vectors)
#1 - pairwise_distances(df.T.to_numpy(), metric='jaccard')
#define Jaccard Similarity function
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

for  i in range(len(vectors)):
    elem = []
    #sample_id.append(i)
    for j in range(len(vectors)):
        i_jac = jaccard(vectors[i], vectors[j])
        elem.append(i_jac)
    print(elem)
    lig = ' '.join(map(str, elem))
    df2 = df2.append({'Id': i,  'Jaccard': lig}, ignore_index=True)
    #Ind_jaccard.append(elem)
#print(vectors)
    #df2 = df2.append({'Jaccard': elem}, ignore_index=True)


df2.to_csv('jacx.csv',sep=' ', index = False)
   