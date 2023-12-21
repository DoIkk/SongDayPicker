# Data Collection & Preprocessing
<details>
<summary>필요한 모듈 설치</summary>

```bash
pip install spotipy
pip install pandas
pip install sqlalchemy
pip install pymysql
pip install python-dotenv
```
</details>
<details>
<summary>모든 장르 종류 받아오기</summary>

rest api로 토큰을 통해 발급받음
[https://api.spotify.com/v1/recommendations/available-genre-seeds](https://api.spotify.com/v1/recommendations/available-genre-seeds)
</details>
<details>
<summary>db 스키마 생성(music 테이블) - mysql</summary>

```sql
create table music;
```
</details>
<details>
<summary>데이터 수집 실행(data_import.py)</summary>
    
```python
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
load_dotenv()
client_id = os.environ.get('client_id')
client_secret = os.environ.get('client_secret')
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

from sqlalchemy import create_engine
import pymysql
db_username = 'root'
db_password = '0000'
db_host = 'localhost'
db_name = 'music'
db_connection_str = f'mysql+pymysql://{db_username}:{db_password}@{db_host}/{db_name}'
db_connection = create_engine(db_connection_str)
conn = db_connection.connect()
genres = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass', 'blues', 'bossanova', 'brazil', 'breakbeat', 'british', 'cantopop', 'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'holidays', 'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 'malay', 'mandopop', 'metal', 'metal-misc', 'metalcore', 'minimal-techno', 'movies', 'mpb', 'new-age', 'new-release', 'opera', 'pagode', 'party', 'philippines-opm', 'piano', 'pop', 'pop-film', 'post-dubstep', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'rainy-day', 'reggae', 'reggaeton', 'road-trip', 'rock', 'rock-n-roll', 'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 'spanish', 'study', 'summer', 'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'trip-hop', 'turkish', 'work-out', 'world-music']

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
# genre_index = int(input("genre's index:"))
year = '2020'
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
    track_features = []
    print(f"year:{year}, genre:{genre}, genre's index: {repeat_index} : searching audio_features...")
    for t_id in track_df['track_id']:
        af = backoff(repeat_index, sp.audio_features,t_id)
        track_features.append(af)
    
    tf_df = pd.DataFrame(columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'url', 'track_href', 'analysis_url', 'duration_ms', 'time_signature'])
    for item in track_features:
        for feat in item:
            tf_df = pd.concat([tf_df,pd.DataFrame(feat, index=[0])], ignore_index=True)
    tf_df.drop(['track_href', 'analysis_url','uri','url','type'], axis=1, inplace=True)  # inplace=True는 원본 데이터프레임을 변경합니다.
    track_df.drop(['artist_genres'],axis=1,inplace=True)
    merged_df = pd.merge(track_df, tf_df, left_on='track_id', right_on='id')
    merged_df.drop(['track_id','artist_id','id'], axis=1, inplace=True)
    merged_df['genre'] = genre
    merged_df.to_sql(name='track', con=db_connection, if_exists='append',index=False)
print(f'{year} done')
```
</details>
<details>
<summary>발매일 상세 정보 없는 데이터 삭제</summary>
    
데이터중에 발매일이 yyyy-mm-dd말고 yyyy만 있는 데이터들이 몇개 있었음

```sql
# 안전 업데이트 모드 잠깐 해제
set sql_safe_updates=0;

DELETE FROM music.track
WHERE LENGTH(release_date) <= 5;

# 안전 업데이트 모드 다시 설정
SET sql_safe_updates=1;
```
</details>
<details>
<summary>날짜 데이터 직렬화</summary>
    
```sql
# 안전 업데이트 모드 잠깐 해제
set sql_safe_updates=0;

# yyyy로만 되어있는 데이터들이 있어서 그런경우 삭제
DELETE from music.track
WHERE LENGTH(release_date) <= 5;

# 이거는 굳이 필요없는거같긴 한데 안되면 해봐요
UPDATE music.track
SET release_date = STR_TO_DATE(release_date, '%Y-%m-%d');

# 날짜 차이 저장할 속성 생성
ALTER TABLE music.track
ADD COLUMN day_difference INT;

# 날짜 차이 저장, 2013-01-01 과의 차이만큼
UPDATE music.track
SET day_difference = DATEDIFF(release_date, '2013-01-01');

# 안전 업데이트 모드 다시 설정
SET sql_safe_updates=1;
```

⇒ 이후 계획이 수정되어 연도 데이터를 뺌

```sql
# 안전 업데이트 모드 잠깐 해제
set sql_safe_updates=0;

#연도 값으로 나눠줌
UPDATE music.track
SET day_difference = day_difference % 365;

# 안전 업데이트 모드 다시 설정
SET sql_safe_updates=1;
```

⇒ 윤년때매 다 꼬여버렸네;;

```sql
# 안전 업데이트 모드 잠깐 해제
set sql_safe_updates=0;

# 달, 일로 수정
ALTER TABLE music.track
ADD COLUMN `month` INT,
ADD COLUMN `day` INT;

UPDATE music.track
SET `month` = MONTH(release_date),
    `day` = DAY(release_date);

# 안전 업데이트 모드 다시 설정
SET sql_safe_updates=1;
```
</details>
<details>  
<summary>장르 인덱스화(genre_to_index.py)</summary>

```python
genres = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass', 'blues', 'bossanova', 'brazil', 'breakbeat', 'british', 'cantopop', 'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'holidays', 'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 'malay', 'mandopop', 'metal', 'metal-misc', 'metalcore', 'minimal-techno', 'movies', 'mpb', 'new-age', 'new-release', 'opera', 'pagode', 'party', 'philippines-opm', 'piano', 'pop', 'pop-film', 'post-dubstep', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'rainy-day', 'reggae', 'reggaeton', 'road-trip', 'rock', 'rock-n-roll', 'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 'spanish', 'study', 'summer', 'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'trip-hop', 'turkish', 'work-out', 'world-music']
import pymysql
db_username = 'root'
db_password = '0000'
db_host = 'localhost'
db_name = 'music'
connection = pymysql.connect(
    host=db_host,
    user=db_username,
    password=db_password,
    database=db_name
)
cursor = connection.cursor()

new_column_name = 'genre_index'
data_type = 'INT'
alter_query = f"ALTER TABLE track ADD {new_column_name} {data_type};"
cursor.execute(alter_query)
print("genre_index 추가")
for index, genre in enumerate(genres, start=1):
    insert_query = f"UPDATE track SET {new_column_name} = {index} WHERE genre = '{genre}';"
    cursor.execute(insert_query)
connection.commit()
```
</details>

# 