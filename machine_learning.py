import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from sqlalchemy import create_engine, text
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

#loss와 test_loss를 저장할 리스트
loss = []
test_loss_plt = []
# Spotify API 설정
client_credentials_manager = SpotifyClientCredentials(client_id='db23c4b3c8b048a8b3408164afda7572', client_secret='927d778a69ad4cdcb0255ebe55edd33f')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# MySQL 연결 설정
db_username = 'root'
db_password = '0000'
db_host = 'localhost'
db_name = 'music'
db_connection_str = f'mysql+pymysql://{db_username}:{db_password}@{db_host}/{db_name}'
db_connection = create_engine(db_connection_str)

# SQL 쿼리 작성 
sql_query = text("SELECT * FROM music.TRACK;")

# 데이터베이스 연결 및 쿼리 실행
with db_connection.connect() as conn:
    result = conn.execute(sql_query)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

# 필요없는 열 drop
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

# trackpopularity를 y로
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

# MLP 모델 정의
class MLP_first(nn.Module):
    # MLP 모델 설계
    def __init__(self):
        # 기본적모델을 처음 학습시켜보고 과적합이 일어나면 dropout과 batch normalization을 적용하여 과적합을 방지
        super(MLP_first, self).__init__()
        self.d1 = nn.Linear(18, 256)
        self.bn1 = nn.BatchNorm1d(256)#batch normalization 적용
        self.dropout1 = nn.Dropout(0.5)#dropout 적용
        self.d2 = nn.Linear(256, 100)
        self.bn2 = nn.BatchNorm1d(100)#batch normalization 적용
        self.dropout2 = nn.Dropout(0.5)#dropout 적용
        self.d3 = nn.Linear(100, 100)
        self.bn3 = nn.BatchNorm1d(100)# batch normalization 적용
        self.dropout3 = nn.Dropout(0.5)#dropout 적용
        self.d4 = nn.Linear(100, 20)
        self.bn4 = nn.BatchNorm1d(20)#batch normalization 적용
        self.dropout4 = nn.Dropout(0.5)#dropout 적용
        self.d5 = nn.Linear(20, 1)
    # 순전파 함수 정의
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
        return x 

# model 생성
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP_first().to(device)

# 손실함수 mse로 설정
criterion = nn.MSELoss()

# optimizer adam으로 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 학습
EPOCHS = 40#epoch 40으로 설정
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

    # 테스트 데이터에 대한 손실함수 값 계산
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(batch_x)
            #test loss 계산
            test_loss = criterion(prediction, batch_y) 
            #test accuracy 계산
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
    test_loss_plt.append(test_loss_avg.item()*4) #test loss는 train loss와 비교하기위해 4배로 설정

# 학습된 모델로 모든 x_train 데이터에 대한 예측값 계산
with torch.no_grad():
    model.eval()  # 모델을 평가 모드로 설정
    predicted_values = model(x_train.to(device)).cpu().numpy().flatten()

# 예측값과 실제값의 차이 계산
difference_values = [pred - (actual * column_stds[0] + column_means[0]) for pred, actual in zip(predicted_values, df_tf[:, y_column_index].numpy())]

#test loss,train loss 그래프 그리기
plt.plot(range(1,len(loss)+1),loss, marker='o', linestyle='-', color='green')
plt.plot(range(1,len(test_loss_plt)+1),test_loss_plt, marker='o', linestyle='-', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.show()

# 모델 저장
torch.save(model.state_dict(), 'model_first.pth')
new_model = MLP_first()
new_model.load_state_dict(torch.load('model_first.pth'))
new_model.eval()

# 예측된 값과 해당하는 월, 일 정보를 저장할 리스트
predicted_values = []
predicted_values2 = []
dates = []
dates2 = []

# 1월 1일부터 12월 31일까지 반복
for month in range(1, 13):
    days_in_month = 31 if month in [1, 3, 5, 7, 8, 10, 12] else 30 if month in [4, 6, 9, 11] else 28  # 각 월별 일 수
    for day in range(1, days_in_month + 1):
        try:
            new_music_data2 = torch.tensor([[81, 16597813, 0.697, 0.875, 5, -4.621, 1, 0.034, 0.137, 0, 0.28, 0.774, 140.069, 163855, 4, 86, month, day]], dtype=torch.float32)
            new_music_data = torch.tensor([[60, 1811721, 0.676, 0.461, 1, -6.746, 0, 0.143, 0.0322, 1.01e-06, 0.358, 0.715, 87.917, 230667, 4, 1, month, day]], dtype=torch.float32)
            new_music_data = new_music_data.view(1, -1)
            normalized_new_data = (new_music_data - column_means[1:]) / column_stds[1:]
            normalized_new_data = normalized_new_data.float()
            new_music_data2 = new_music_data2.view(1, -1)
            normalized_new_data2 = (new_music_data2 - column_means[1:]) / column_stds[1:]
            normalized_new_data2 = normalized_new_data2.float()
            # 예측값 계산
            with torch.no_grad():
                predicted_value = new_model(normalized_new_data)
                predicted_values.append(predicted_value.item())
                dates.append(f"{month:02d}-{day:02d}")  # 월과 일 정보만 저장 (mm-dd 형식)
                predicted_value2 = new_model(normalized_new_data2)
                predicted_values2.append(predicted_value2.item())
                dates2.append(f"{month:02d}-{day:02d}")

        except ValueError:
            # 유효하지 않은 날짜의 경우 패스
            print(f"Invalid date: {month:02d}-{day:02d}")
            continue

predicted_values_plt = []
# 예측된 값과 해당 월, 일 정보 출력
for idx, (value, date) in enumerate(zip(predicted_values, dates)):
    print(f"Processing Date: {date}")
    predicted_values_plt.append( value * column_stds[0] + column_means[0] )
    print(f"Predicted Value: {value * column_stds[0] + column_means[0]}")

predicted_values_plt2 = []
# 예측된 값과 해당 월, 일 정보 출력
for idx, (value, date) in enumerate(zip(predicted_values2, dates2)):
    print(f"Processing Date2: {date}")
    predicted_values_plt2.append( value * column_stds[0] + column_means[0] )
    print(f"Predicted Value2: {value * column_stds[0] + column_means[0]}")
    
#날짜에 따른 인기도 그래프 그리기
plt.plot(range(1,len(predicted_values_plt)+1),predicted_values_plt, marker='o', linestyle='-', color='green')
plt.plot(range(1,len(predicted_values_plt2)+1),predicted_values_plt2, marker='o', linestyle='-', color='blue')
plt.xlabel('Date')
plt.ylabel('Popularity')
plt.title('Predicted Graph')
plt.show()
