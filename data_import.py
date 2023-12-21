import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

# api키 숨기기 위해 .env파일에서 키 받아옴
load_dotenv()
client_id = os.environ.get('client_id')
client_secret = os.environ.get('client_secret')
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
from sqlalchemy import create_engine
import pymysql

# DB정보 입력
db_username = 'root'
db_password = '0000'
db_host = 'localhost'
db_name = 'music'
db_connection_str = f'mysql+pymysql://{db_username}:{db_password}@{db_host}/{db_name}'
db_connection = create_engine(db_connection_str)
conn = db_connection.connect()

# 장르 종류
genres = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass', 'blues', 'bossanova', 'brazil', 'breakbeat', 'british', 'cantopop', 'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'holidays', 'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 'malay', 'mandopop', 'metal', 'metal-misc', 'metalcore', 'minimal-techno', 'movies', 'mpb', 'new-age', 'new-release', 'opera', 'pagode', 'party', 'philippines-opm', 'piano', 'pop', 'pop-film', 'post-dubstep', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'rainy-day', 'reggae', 'reggaeton', 'road-trip', 'rock', 'rock-n-roll', 'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 'spanish', 'study', 'summer', 'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'trip-hop', 'turkish', 'work-out', 'world-music']

# backoff 알고리즘
import time
def backoff(index,request_function, *args, **kwargs):
    retries = 10  # Number of retries
    for i in range(retries):
        try:
            response = request_function(*args, **kwargs)
            return response
        except Exception as e:
            print(f'An error occurred in genre: {genres[index]}, index:{index}.')
            print(f"Retrying in 20 minutes…")
            time.sleep(1200)
    return None

# 데이터 수집 
# genre_index = int(input("genre's index:"))
year = '2020'

# 전에 받은 장르부터 시작 할 수 있도록
query = "SELECT DISTINCT genre FROM test.track WHERE release_date LIKE %(year)s"
data = pd.read_sql(query, conn, params={'year': year + '%'})
if(len(data)==0):
    genre_index = 0
else:
    genre_index = genres.index(data.iloc[-1].genre)
for repeat_index,genre in enumerate(genres):
    if repeat_index<genre_index:
        continue
    artist_name =[]
    track_name = []
    track_popularity =[]
    artist_id =[]
    track_id =[]
    track_release = []

    # 트랙 정보 받아오기
    print(f"year:{year}, genre:{genre}, genre's index: {repeat_index} : searching track...")
    track_results = backoff(repeat_index, sp.search,q=f'year:{year} genre:{genre}', type='track', limit=50, offset=0)
    if(track_results['tracks']['total']==0):
        print(f"lack of data in {genre}")
        continue
    for i, t in enumerate(track_results['tracks']['items']):
        artist_name.append(t['artists'][0]['name'])
        artist_id.append(t['artists'][0]['id'])
        track_name.append(t['name'])
        track_id.append(t['id'])
        track_popularity.append(t['popularity'])
        track_release.append(t['album']['release_date'])
    track_df = pd.DataFrame({'artist_name' : artist_name, 'track_name' : track_name, 'track_id' : track_id, 'track_popularity' : track_popularity, 'artist_id' : artist_id, 'release_date': track_release})
    
    #아티스트 정보 받아오기
    artist_popularity = []
    artist_genres = []
    artist_followers =[]
    print(f"year:{year}, genre:{genre}, genre's index: {repeat_index} : searching artist...")
    for a_id in track_df.artist_id:
        artist = backoff(repeat_index, sp.artist,a_id)
        artist_popularity.append(artist['popularity'])
        artist_genres.append(artist['genres'])
        artist_followers.append(artist['followers']['total'])
    track_df = track_df.assign(artist_popularity=artist_popularity, artist_genres=artist_genres, artist_followers=artist_followers)
    
    #audio feature 받아오기
    track_features = []
    print(f"year:{year}, genre:{genre}, genre's index: {repeat_index} : searching audio_features...")
    for t_id in track_df['track_id']:
        af = backoff(repeat_index, sp.audio_features,t_id)
        track_features.append(af)
    tf_df = pd.DataFrame(columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'url', 'track_href', 'analysis_url', 'duration_ms', 'time_signature'])
    for item in track_features:
        for feat in item:
            tf_df = pd.concat([tf_df,pd.DataFrame(feat, index=[0])], ignore_index=True)

    #간단한 데이터 전처리
    tf_df.drop(['track_href', 'analysis_url','uri','url','type'], axis=1, inplace=True)
    track_df.drop(['artist_genres'],axis=1,inplace=True)
    merged_df = pd.merge(track_df, tf_df, left_on='track_id', right_on='id')
    merged_df.drop(['track_id','artist_id','id'], axis=1, inplace=True)
    merged_df['genre'] = genre

    #데이터 mysql로 보냄
    merged_df.to_sql(name='track', con=db_connection, if_exists='append',index=False)

print(f'{year} done')