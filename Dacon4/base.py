# 라이브러리 임포트
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.neighbors import KNeighborsRegressor  # 사용할 모델입니다.

# 1차 데이터 불러오기
# fn = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data/train.csv')

# 2차 데이터 불러오기
fn = pd.read_csv('C:/data/Dacon/data_v2/train_v2.csv')
fn.tail()


# 데이터 불러오기
# 2000년도 데이터부터 불러옵니다.
dm = fn.iloc[255:,1]
dm.shape

# 데이터 생성
# 2000년도 이후 데이터들을 불러와 하나의 데이터 셋으로 만듭니다.
data = np.load('C:/data/Dacon/data_v2/train_data_v2/200001.npy')
data = data.reshape(1,448,304,5)

for i in tqdm(dm):
    a = np.load('C:/data/Dacon/data_v2/train_data_v2/'+i)
    a = a.reshape(1,448,304,5)

    data = np.concatenate((data,a), axis=0)
data = np.array(data)
data = data[:,:,:,0]

data.shape


# 월별 데이터 셋 만들기
# 월별로 데이터를 뽑아 합치는 For문입니다.

# 데이터 셋 형태 변환
data = data.reshape(240,1, 448,304)

# 월별로 데이터 셋 생성
for i in tqdm(range(12)):
  globals()['train{}'.format(i)] = np.array(np.concatenate((data[0+i], data[12+i], data[24+i], data[36+i], data[48+i], data[60+i], data[72+i], data[84+i], data[96+i], data[108+i], data[120+i],
                                                            data[132+i], data[144+i], data[156+i], data[168+i], data[180+i], data[192+i], data[204+i], data[216+i], data[228+i]), axis=0))

  print("\n",globals()['train{}'.format(i)].shape)  # globals()는 그 변수를 의미 - 없으면 그냥 문자열



# 그림을 그리는 함수
# 코드공유 - 'DATA loading + Simple EDA + 참고가능 논문' 내의 함수를 약간 변경하였습니다.
# 코드 공유를 해주신 Jay윤님 감사합니다.

def show(npy):
    num_channel = npy.shape[0]
    plt.figure(figsize=(50, 50)) 
    for channel in range(num_channel):
      tmpimg = npy[channel, :, :]
      ax = plt.subplot(1, num_channel, channel+1)  # (행, 열, 데이터 개수) - 그리고 싶은 팜플렛 모양
      ax.title.set_text("Ice")

      ax.imshow(tmpimg)
    plt.tight_layout()
    plt.show()
    plt.close()

# 소숫점 둘째 자리에서 반올림하는 함수

def fun1(x) : 
  return np.around(x,2)


# 점수 계산
# 대회안내 - 규칙에 존재하는 산식 코드입니다.

def mae_score(true, pred):
    score = np.mean(np.abs(true-pred))
    
    return score

def f1_score(true, pred):
    target = np.where((true>250*0.05)<250*0.5)
    
    true = true[target]
    pred = pred[target]
    true = np.where(true < 250*0.15, 0, 1)
    pred = np.where(pred < 250*0.15, 0, 1)
    
    right = np.sum(true * pred == 1)
    precision = right / np.sum(true+1e-8)
    recall = right / np.sum(pred+1e-8)
    score = 2 * precision*recall/(precision+recall+1e-8)
    
    return score
    
def mae_over_f1(true, pred):
    mae = mae_score(true, pred)
    f1 = f1_score(true, pred)
    score = mae/(f1+1e-8)
    
    return score


# 이미지 EDA (날짜는 오름차순입니다. '2000->2001->2002...')
# 가운데에 빈 공간이 작은 12개까지를 기준 훈련데이터 양으로 잡았습니다.
# 12개가 기준인 이유는 크게 없습니다. 
# 그저 EDA를 통해서 가운데 빈공간이 작은 것들이 예측에 도움이 될 것이라 생각했습니다.
# 실제로 빈공간이 있는 데이터들까지 훈련하면 예측된 값들도 가운데 빈공간이 큽니다.(예측값의 손실이 발생)

show(train0)


# n_neighbors의 값을 찾기 위한 노력 1

# 훈련데이터 12개만
x_train = train10[5:17].reshape( 12,-1).T
y_train = train10[17].reshape(1,-1).T  # 변동없음
x_test = train10[6:18].reshape( 12,-1).T
real = train10[18]  # 변동없음

# K값을 찾아 1
from sklearn.neighbors import KNeighborsRegressor

num = [610, 620]

for i in tqdm(num):
  print(i)
  model = KNeighborsRegressor(n_neighbors= i, weights='distance', p=1, n_jobs=-1)
  model.fit(x_train, y_train)

  pre = model.predict(x_test)
  pre = pre.reshape(-1)
  
  print( "MAE : %s" % mae_score(real.reshape(-1), fun1(pre)),
      '\n F1 : %s' % f1_score(real.reshape(-1), fun1(pre)),
      '\n Final %s' % mae_over_f1(real.reshape(-1), fun1(pre)))


# n_neighbors의 값을 찾기 위한 노력 2

for j in tqdm(range(12)):
  print(j)
  dataset = globals()['train{}'.format(j)]

  # 훈련데이터 12개만
  x_train = dataset[6:18].reshape( 12,-1).T
  y_train = dataset[18].reshape(1,-1).T  # 변동없음
  x_test = dataset[7:19].reshape( 12,-1).T
  real = dataset[19]  # 변동없음

  # K값을 찾아 2
  from sklearn.neighbors import KNeighborsRegressor

  num = [500, 550, 600, 601, 650, 700, 750, 800, 850, 900, 950]

  for i in num:
    print(i)
    model = KNeighborsRegressor(n_neighbors= i, weights='distance', p=1, n_jobs=-1)
    model.fit(x_train, y_train)

    pre = model.predict(x_test)
    pre = pre.reshape(-1)
  
    print( "MAE : %s" % mae_score(real.reshape(-1), fun1(pre)),
           '\n F1 : %s' % f1_score(real.reshape(-1), fun1(pre)),
           '\n Final %s' % mae_over_f1(real.reshape(-1), fun1(pre)))


# n_neighbors의 값을 찾기 위한 노력 3

# 훈련데이터 12개만
x_train = train1[6:18].reshape( 12,-1).T
y_train = train1[18].reshape(1,-1).T  # 변동없음
x_test = train1[7:19].reshape( 12,-1).T
real = train1[19]  # 변동없음

# K값을 찾아 3
from sklearn.neighbors import KNeighborsRegressor

num = [580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593]

for i in tqdm(num):
  print(i)
  model = KNeighborsRegressor(n_neighbors= i, weights='distance', p=1, n_jobs=-1)
  model.fit(x_train, y_train)

  pre = model.predict(x_test)
  pre = pre.reshape(-1)
  
  print( "MAE : %s" % mae_score(real.reshape(-1), fun1(pre)),
      '\n F1 : %s' % f1_score(real.reshape(-1), fun1(pre)),
      '\n Final %s' % mae_over_f1(real.reshape(-1), fun1(pre)))


# n_neighbors의 값을 찾기 위한 노력 4

# 훈련데이터 12개만
x_train = train1[6:18].reshape( 12,-1).T
y_train = train1[18].reshape(1,-1).T  # 변동없음
x_test = train1[7:19].reshape( 12,-1).T
real = train1[19]  # 변동없음

# K값을 찾아 4
from sklearn.neighbors import KNeighborsRegressor

num = [594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605]

for i in tqdm(num):
  print(i)
  model = KNeighborsRegressor(n_neighbors= i, weights='distance', p=1, n_jobs=-1)
  model.fit(x_train, y_train)

  pre = model.predict(x_test)
  pre = pre.reshape(-1)
  
  print( "MAE : %s" % mae_score(real.reshape(-1), fun1(pre)),
      '\n F1 : %s' % f1_score(real.reshape(-1), fun1(pre)),
      '\n Final %s' % mae_over_f1(real.reshape(-1), fun1(pre)))

# 배포된 2차 데이터
# 2019년 데이터 가져오기
dm1 = fn.iloc[483:,1]
dm1.shape


# 2019년 데이터 전처리
data1 = np.load('C:/data/Dacon/data_v2/train_data_v2/201901.npy')
data1 = data1.reshape(1,448,304,5)

for i in tqdm(dm1):
    a = np.load('C:/data/Dacon/data_v2/train_data_v2/'+i)
    a = a.reshape(1,448,304,5)

    data1 = np.concatenate((data1,a), axis=0)
data1 = np.array(data1)
data1 = data1[:,:,:,0]
sh1 = data1.reshape(12, -1)
sh1 = pd.DataFrame(sh1)

sh1.shape


# 기본
from sklearn.neighbors import KNeighborsRegressor

for i in tqdm(range(12)):
  datan = globals()['train{}'.format(i)]

  # 데이터셋 나누기
  x_train = datan[7:19].reshape(12,-1).T
  y_train = datan[19].reshape(1,-1).T
  x_test = datan[8:20].reshape(12,-1).T

  # 모델 훈련
  model = KNeighborsRegressor(n_neighbors=601, weights='distance', p=1, n_jobs=-1) #601
  model.fit(x_train, y_train)

  # 모델 예측
  predictions = model.predict(x_test)
  predictions = predictions.reshape(-1)

  # 그래프를 보자구
  plt.imshow(predictions.reshape(448,304), interpolation = 'None')
  plt.show()

  # 변수 저장
  globals()['pred_{}'.format(i)] = predictions
  print(globals()['pred_{}'.format(i)].shape)


  # 제출 형태로 변환

# 데이터 생성
sh2 = pd.DataFrame(pred_0.reshape(1,-1))  # 초기값 설정
sh2 = fun1(sh2)  # 음수와 소숫점 없애기

for i in range(11):
  globals()['pred_{}'.format(i+1)] = globals()['pred_{}'.format(i+1)].reshape(1,-1)

  sh = globals()['pred_{}'.format(i+1)]
  sh = fun1(sh)  # 음수와 소숫점 없애기
  sh = pd.DataFrame(sh)

  sh2 = pd.concat((sh2, sh), axis=0)
  print(sh2.shape)


  # 제출 데이터 인덱스 수정

# 2019 데이터와 2020 예측 합치기
result = pd.concat([sh1,sh2], axis=0)
result = result.reset_index(drop=True)

# 제출 파일과 결합
submission = pd.read_csv("C:/data/Dacon/data_v2/sample_submission.csv")
sub = pd.concat([submission.loc[:,'month'],result], axis=1)
sub.columns = submission.columns.values  # 제출파일에서 컬럼명을 
sub.tail()


# 2차 데이터 확인 (최종 제출본)
check = np.array(sub.iloc[10,1:], dtype=np.float64).reshape(448,304)

import matplotlib.pyplot
matplotlib.pyplot.imshow(check, interpolation = 'None')
matplotlib.pyplot.title('Lets Final Check')


sub.to_csv('C:/data/Dacon/data_v2/sub_0413_1_KNN(12, 601).csv', index = False)

# 기본
from sklearn.neighbors import KNeighborsRegressor

for i in tqdm(range(12)):
  datan = globals()['train{}'.format(i)]

  # 데이터셋 나누기
  x_train = datan[6:18].reshape(12,-1).T 
  y_train = datan[18].reshape(1,-1).T
  x_test = datan[7:19].reshape(12,-1).T

  # 모델 훈련
  model = KNeighborsRegressor(n_neighbors=601, weights='distance', p=1, n_jobs=-1) 
  model.fit(x_train, y_train)

  # 모델 예측
  predictions = model.predict(x_test)
  predictions = predictions.reshape(-1)

  # 그래프
  plt.imshow(predictions.reshape(448,304), interpolation = 'None')
  plt.show()

  # 변수 저장
  globals()['pred{}'.format(i)] = predictions
  print(globals()['pred{}'.format(i)].shape)