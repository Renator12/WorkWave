import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from datetime import datetime

# collect data from Spotify API and collect into JSON file
client_credentials_manager = SpotifyClientCredentials(client_id='36ce608884584cfeafa3d53bae9908dd', client_secret='6fb04c6eae894b168801e8900bcdbb12')
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

playlist_link = input("Enter the link: ")

playlist_URI = playlist_link.split("/")[-1].split("?")[0]

songs = sp.playlist_items(playlist_URI, market='IN')['items']  #will give a list of songs (meta-data)

song_name = []
song_id = []
song_popu = []
song_added_date = []
song_release_date = []
artists_col = []
for song in songs:
    song_name.append(song['track']['name'])
    song_id.append(song['track']['id'])
    song_popu.append(song['track']['popularity'])
    song_added_date.append(song['added_at'])
    song_release_date.append(song['track']['album']['release_date'])
    all_artists = song['track']['artists']
    artists = []
    for a in all_artists:
        artists.append(a['name'])
    artists_col.append(artists)

# Compile data into pandas dataframe
df = pd.DataFrame({
    'name':song_name,
    'popularity':song_popu,
    'date_added':pd.to_datetime(song_added_date),
    'release_year':list(map(lambda x: int(x[:4]), song_release_date)),
    'artists':artists_col
    })

#audio features

features = sp.audio_features(song_id)

feat_names = list(sp.audio_features(song_id)[0].keys())

for row in range(len(features)):
    for col in range(len(feat_names)):
        df.loc[row, feat_names[col]] = features[row][feat_names[col]]
df.head()

# get updated songs as playlist is updated
curr_month = datetime.today().month
curr_year = datetime.today().year

recency = list(map(lambda x: curr_month - x.month if (x.year == curr_year) else curr_month + (12 - x.month)
                    + (curr_year - x.year - 1) * 12, df['date_added']))
df['recency'] = recency

# drop songs with release year 0 (clean data)
for i in range(len(df['release_year'])):
    if df.loc[i, 'release_year'] == 0:
        df.drop(i, inplace=True)

# math
df['popularity'] = list(map(lambda x: x // 5, df['popularity']))

# for cosine similarity, we need the size of the vectors to be the same, so we are genralizing the columns
for i in range(0,21):
    df[f"popu|{i}"] = [0] * len(df['name'])

for i in range(1980, datetime.today().year + 1):
    df[f"year|{i}"] = [0] * len(df['name'])

# this will create a dataframe with the columns of unique values in the series
df_year = pd.get_dummies(df['release_year'])
df_popu = pd.get_dummies(df['popularity'])

# assigning names to the columns
df_year.columns = map(lambda x: 'year' + '|' + str(x), df_year.columns)
df_popu.columns = map(lambda x: 'popu' + '|' + str(x), df_popu.columns)

#now updating the columns with values wherever needed
for col in df_popu.columns:
    df[col] = df_popu[col]
for col in df_year.columns:
    df[col] = df_year[col]

df = pd.DataFrame([])
artists_col = []
for i in range(len(songs)):
    all_artists = songs[i]['track']['artists']
    artists = []
    for a in all_artists:
        artists.append(a['name'])
    artists_col.append(artists)
df['artists'] = artists_col

df = df.explode('artists').drop_duplicates().reset_index().drop('index', axis=1)

df1 = pd.read_csv("datasets/artists_names.csv")

df1.drop('Unnamed: 0', axis=1, inplace=True)

df1 = pd.concat([df1, df]).drop_duplicates().reset_index().drop('index', axis=1)

# how do I work with csvs????
df1.to_csv('C:\Users\smita\Desktop\Serenity Hacks\artists_names.csv')

# this file contains artists' names which will be used for ohe
artists_csv = pd.read_csv('datasets/artists_names.csv')

# creating dummy dataframe for ohe-ing the artists
zeros = [0] * len(df['name'])
extra = pd.DataFrame(zeros)
for name in artists_csv['artists']:
    extra[f"artist|{name}"] = 0

new_df = pd.concat([df, extra], axis=1)
new_df.dropna(axis=0, inplace=True)

# to place 1 whenever the artist in the row cell matches with the column artist
for i, row in new_df.iterrows():
    for name in row['artists']:
        if name in list(artists_csv['artists']):
            new_df.loc[i, f"artist|{name}"] = 1

new_df = new_df.drop(0, axis=1)

'''
new df for generating the recommendation vector
we are dropping the non-integer columns as they are of no use in calculating the similarity.
Also, dropping the ones which will not help in recomme
'''
recomm_vec_df = new_df.drop(['name', 'popularity', 'date_added', 'release_year', 'type', 'id', 'uri', 'track_href',  'analysis_url', 'artists', 'key', 'mode', 'duration_ms', 'time_signature'], axis=1)

recomm_vec_df['bias'] = list(map(lambda x: round(0.9 ** x, 5), list(recomm_vec_df['recency'])))
print(recomm_vec_df.loc[:,'bias'])
for col in recomm_vec_df.columns[14:]:
    recomm_vec_df[col] = recomm_vec_df[col] * recomm_vec_df['bias']

recomm_vec_df = recomm_vec_df.dropna().drop(['bias', 'recency'], axis=1)

recomm_vec_df['tempo'] = recomm_vec_df['tempo'].apply(lambda x: (x - min(recomm_vec_df['tempo'])) / (max(recomm_vec_df['tempo'] - min(recomm_vec_df['tempo']))))
recomm_vec_df['loudness'] = recomm_vec_df['loudness'].apply(lambda x: (x - min(recomm_vec_df['loudness'])) / (max(recomm_vec_df['loudness'] - min(recomm_vec_df['loudness']))))

# this one will create the song features columns vector
recomm_vec1 = np.array(list(map(lambda col: recomm_vec_df[col].mean(), recomm_vec_df.loc[:, :"tempo"].columns)))
# this one will create the ohe columns till current year vector
recomm_vec2 = np.array(list(map(lambda col: sum(recomm_vec_df[col]), recomm_vec_df.loc[:, "popu|0":f"year|{datetime.today().year}"].columns)))
# artists only ohe columns vector
recomm_vec3 = np.array(list(map(lambda col: sum(recomm_vec_df[col]), recomm_vec_df.iloc[:, -len(artists_excel['artists']):].columns)))

data = pd.read_csv('datasets/final_data.csv')

# columns that will be used for the filtering
filt_col = ['acousticness', 'danceability', 'energy', 'loudness', 'tempo', 'valence']

# values for filtering (Emotion specific)
happy_low = [0, 0.57, 0.4, -10.4, 75, 0.25]
sad_low = [0.2, 0.3, 0.25, -11, 70, 0]
chill_low = [0, 0.35, 0.25, -12.7, 80, 0.2]
angry_low = [0, 0.46, 0.56, -11, 90, 0.2]

happy_high = [0.75, 0.86, 1, -3, 170, 1]
sad_high = [0.9, 0.7, 0.8, -4, 160, 0.7]
chill_high = [0.85, 0.8, 0.8, -4, 165, 0.9]
angry_high = [0.6, 0.85, 1, -4, 170, 0.75]

happy_avg = [0.715, 0.7, 0.375, -6.7, 0.625, 123]
sad_avg = [0.5, 0.525, 0.55, -7.5, 0.3, 115]
chill_avg = [0.575, 0.525, 0.425, -8.35, 0.55, 122.5]
angry_avg = [0.655, 0.78, 0.3, -7.5, 0.475, 130]

# e: song's feature values

i = 0
if emotion == 'happy':
    for col in filt_col:
        data = data[(data[col] > happy_low[i]) & (data[col] < happy_high[i])]
        i += 1

    sim = []
    for i in range(len(data)):
        e = data.loc[:, filt_col].iloc[i].values
        sim.append(np.linalg.norm(e - happy_avg) / 70)
    data['sim'] = (np.array(sim) - max(sim)) * (-1)

elif emotion == 'sad':
    for col in filt_col:
        data = data[(data[col] > sad_low[i]) & (data[col] < sad_high[i])]
        i += 1

    sim = []
    for i in range(len(data)):
        e = data.loc[:, filt_col].iloc[i].values
        sim.append(np.linalg.norm(e - sad_avg) / 70)
    data['sim'] = (np.array(sim) - max(sim)) * (-1)

elif emotion == 'neutral':
    for col in filt_col:
        data = data[(data[col] > chill_low[i]) & (data[col] < chill_high[i])]
        i += 1

    sim = []
    for i in range(len(data)):
        e = data.loc[:, filt_col].iloc[i].values
        sim.append(np.linalg.norm(e - chill_avg) / 70)
    data['sim'] = (np.array(sim) - max(sim)) * (-1)

elif emotion == 'angry':
    for col in filt_col:
        data = data[(data[col] > angry_low[i]) & (data[col] < angry_high[i])]
        i += 1

    sim = []
    for i in range(len(data)):
        e = data.loc[:, filt_col].iloc[i].values
        sim.append(np.linalg.norm(e - angry_avg) / 70)
    data['sim'] = (np.array(sim) - max(sim)) * (-1)
    data_filtered = data.drop(
        ['name', 'popularity', 'date_added', 'release_year', 'type', 'id', 'uri', 'track_href', 'analysis_url',
         'artists', 'Unnamed: 0', 'key', 'mode', 'duration_ms', 'time_signature'], axis=1)

    # normalizing the tempo and loudness in the main dataset
    data_filtered['tempo'] = data_filtered['tempo'].apply(
        lambda x: (x - min(data_filtered['tempo'])) / (max(data_filtered['tempo'] - min(data_filtered['tempo']))))
    data_filtered['loudness'] = data_filtered['loudness'].apply(lambda x: (x - min(data_filtered['loudness'])) / (
        max(data_filtered['loudness'] - min(data_filtered['loudness']))))

    '''
    Using Euclidian Distance as both magnitude and direction is important
    Euclidean distance measures the distance between two points in a multidimensional space by calculating the square
    root of the sum of the squared differences between their corresponding elements. It is suitable for continuous data
    where the magnitude and direction of each feature are important.
    '''

    l1 = []
    l2 = []
    l3 = []
    s = 0
    recommendations = pd.DataFrame(
        {'name': data['name'], 'artists': data['artists'], 'id': data['id'], 'sim': data['sim']})
    for i in range(len(data_filtered)):
        # this contains the columns from the start till the ohe
        data_1 = data_filtered.loc[:, :"tempo"].iloc[i].values
        # this contains the ohe columns till the current year
        data_2 = data_filtered.loc[:, "popu|0":f"year|{datetime.today().year}"].iloc[i].values
        # this contains the artists-only columns
        data_3 = data_filtered.iloc[:, (-len(artists_excel['artists']) - 1):-1].iloc[i].values

        sim1 = np.linalg.norm(recomm_vec1 - data_1)  # euclidian distance
        '''
        we are getting a dissimilarity score, as the greater the difference 
        between the values, the higher would be the score. The values which differ largely with respect to the vector
        will tend to have a higher Euclidian score
        '''

        # simply using dot product
        sim2 = np.dot(recomm_vec2, data_2)

        sim3 = np.dot(recomm_vec3, data_3)

        l1.append(round(sim1, 6))
        l2.append(round(sim2, 6))
        l3.append(round(sim3, 6))

    l1 = (np.array(l1) - max(l1)) * (-1)  # converting it into a similarity score

    # normalizing the array values to the 0-1 range for proper contribution in the recommendation
    l2 = (np.array(l2) - min(l2)) / (max(l2) - min(l2))
    l3 = (np.array(l3) - min(l3)) / (max(l3) - min(l3)) * 0.5  # limiting to 0-0.5 range

    score = l1 + l2 + l3

    recommendations['sim'] = recommendations['sim'] + score  # as sim col is already filled with emotion effiency score



