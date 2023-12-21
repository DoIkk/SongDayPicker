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
<details>
<summary>data 전처리</summary>
    
```python
df = df.drop(columns=['release_date', 'track_name', 'artist_name', 'genre','day_difference'])

# 표준화를 위한 scaler 정의
scaler = StandardScaler()

# 데이터 프레임을 넘파이 배열로 변환
numpy_array = df.values
#나중에 예시데이터 표준화를 위한 평균과 표준편차 저장
column_means = np.mean(numpy_array, axis=0)
column_stds = np.std(numpy_array, axis=0)

# 정규화 (표준화)
normalized_array = scaler.fit_transform(numpy_array)

# 정규화된 넘파이 배열을 텐서로 변환
df_tf = torch.tensor(normalized_array, dtype=torch.float32)

# date_difference를 y로
y_column_index = 0
y_data = df_tf[:, y_column_index]
# 나머지 x로
x_data = torch.cat((df_tf[:, :y_column_index], df_tf[:, y_column_index + 1:]), dim=1)

# train과 test로 분할
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# 배치 사이즈 설정
batch_size = 500
# train 데이터셋과 DataLoader 정의
train_data = TensorDataset(x_train, y_train.view(-1, 1))  # 정답 데이터를 2차원으로 변환
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# test 데이터셋과 DataLoader 정의
test_data = TensorDataset(x_test, y_test.view(-1, 1))  # 정답 데이터를 2차원으로 변환
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
```
</details>


# model generater
<details>
<summary>필요한 모듈 설치</summary>
    
```python
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from sqlalchemy import create_engine, text
```
</details>
<details>
<summary>MLP모델 설계(첫번째 모델: 날짜를 뺀 피쳐들로 부터 인기도 예측모델)</summary>
    
```python
class MLP_first(nn.Module):
    def __init__(self):
        super(MLP_first, self).__init__()
        self.d1 = nn.Linear(18, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.d2 = nn.Linear(256, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.dropout2 = nn.Dropout(0.5)
        self.d3 = nn.Linear(100, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.dropout3 = nn.Dropout(0.5)
        self.d4 = nn.Linear(100, 20)
        self.bn4 = nn.BatchNorm1d(20)
        self.dropout4 = nn.Dropout(0.5)
        self.d5 = nn.Linear(20, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.d1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.d2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.d3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn4(self.d4(x)))
        x = self.dropout4(x)
        x = self.d5(x)
        # x = F.relu(self.d4(x))
        return x
```
</details>
<details>
<summary>모델생성 및 학습</summary>
    
```python
# model 생성
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP_first().to(device)

# 손실함수f
criterion = nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    train_loss_avg = 0
    train_acc_avg = 0
    train_num = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        prediction = model(batch_x)
        train_loss = criterion(prediction, batch_y)
        train_loss.backward()

        y_pred = torch.round(prediction).squeeze()
        train_acc_avg += (y_pred == batch_y.view_as(y_pred)).sum().item()
        train_num += len(batch_y)
        optimizer.step()
        train_loss_avg += train_loss / len(batch_y)
    
    train_acc_avg /= train_num
    test_loss_avg = 0
    test_acc_avg = 0
    test_num = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(batch_x)

            test_loss = criterion(prediction, batch_y) 

            y_pred = torch.round(prediction).squeeze()
            test_acc_avg += (y_pred == batch_y.view_as(y_pred)).sum().item() 
            test_num += len(batch_y)

            test_loss_avg += test_loss / len(batch_y)
    
    test_acc_avg /= test_num

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss_avg}, '
        f'Test Loss: {test_loss_avg}, '
    )
    loss.append(train_loss_avg.item())
    test_loss_plt.append(test_loss_avg.item()*4)
```
</details>
<details>
<summary>모델 성능 평가를 위한 시각화</summary>
        
```python
# 학습된 모델로 모든 x_train 데이터에 대한 예측값 계산
with torch.no_grad():
    model.eval()  # 모델을 평가 모드로 설정
    predicted_values = model(x_train.to(device)).cpu().numpy().flatten()

# 예측값과 실제값의 차이 계산
difference_values = [pred - (actual * column_stds[0] + column_means[0]) for pred, actual in zip(predicted_values, df_tf[:, y_column_index].numpy())]

# Plot the histogram with a solid line
plt.plot(range(1,len(loss)+1),loss, marker='o', linestyle='-', color='green')
plt.plot(range(1,len(test_loss_plt)+1),test_loss_plt, marker='o', linestyle='-', color='orange')
# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Graph')

# Show the plot
plt.show()
```
</details>
<details>
<summary>모델 저장</summary>

```python
torch.save(model.state_dict(), 'model_first.pth')
```
</details>
<details>   
<summary>모델 로드</summary>

```python
# load model
new_model = MLP_first()
new_model.load_state_dict(torch.load('model_first.pth'))
new_model.eval()
```
</details>
<details>     
<summary>다른 발매일에 따른 예측 인기도 생성</summary>
            
```python
# 예측된 값과 해당하는 월, 일 정보를 저장할 리스트
predicted_values = []
dates = []

for month in range(1, 13):
    days_in_month = 31 if month in [1, 3, 5, 7, 8, 10, 12] else 30 if month in [4, 6, 9, 11] else 28  # 각 월별 일 수
    for day in range(1, days_in_month + 1):
        try:
            new_music_data2 = torch.tensor([[]])
            new_music_data = torch.tensor([[60, 1811721, 0.676, 0.461, 1, -6.746, 0, 0.143, 0.0322, 1.01e-06, 0.358, 0.715, 87.917, 230667, 4, 1, month, day]], dtype=torch.float32)
            new_music_data = new_music_data.view(1, -1)
            normalized_new_data = (new_music_data - column_means[1:]) / column_stds[1:]
            normalized_new_data = normalized_new_data.float()

            with torch.no_grad():
                predicted_value = new_model(normalized_new_data)
                predicted_values.append(predicted_value.item())
                dates.append(f"{month:02d}-{day:02d}")  # 월과 일 정보만 저장 (mm-dd 형식)
        except ValueError:
            # 유효하지 않은 날짜의 경우 패스
            print(f"Invalid date: {month:02d}-{day:02d}")
            continue
```
</details>    
<details>
<summary>예측 인기도 출력과 시각화</summary>
                
```python
# 예측된 값과 해당 월, 일 정보 출력
predicted_values_plt = []
for idx, (value, date) in enumerate(zip(predicted_values, dates)):
    print(f"Processing Date: {date}")
    predicted_values_plt.append( value * column_stds[0] + column_means[0] )
    print(f"Predicted Value: {value * column_stds[0] + column_means[0]}")

    # Plot the histogram with a solid line
plt.plot(range(1,len(predicted_values_plt)+1),predicted_values_plt, marker='o', linestyle='-', color='green')
# Add labels and title
plt.xlabel('Date')
plt.ylabel('Popularity')
plt.title('Predicted Graph')

# Show the plot
plt.show()
```
</details>