import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

train = pd.read_csv('../data/csv/train/train.csv')
submission = pd.read_csv('../data/csv/sample_submission.csv')

day = 4 # 예측을 위한 일 수

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

def split_to_seq(data):
    tmp = []
    for i in range(48):
        tmp1 = pd.DataFrame()
        for j in range(int(len(data)/48)):
            tmp2 = data.iloc[j*48+i,:]
            tmp2 = tmp2.to_numpy()
            tmp2 = tmp2.reshape(1,tmp2.shape[0])
            tmp2 = pd.DataFrame(tmp2)
            tmp1 = pd.concat([tmp1,tmp2])
        x = tmp1.to_numpy()
        tmp.append(x)
    return np.array(tmp)
def make_cos(dataframe): # 특정 열이 해가 뜨고 해가지는 시간을 가지고 각 시간의 cos를 계산해주는 함수!!
    dataframe /=dataframe
    c = dataframe.dropna()
    d = c.to_numpy()

    def into_cosine(seq):
        for i in range(len(seq)):
            if i < len(seq)/2:
                seq[i] = float((len(seq)-1)/2) - (i)
            if i >= len(seq)/2:
                seq[i] = seq[len(seq) - i - 1]
        seq = seq/ np.max(seq) * np.pi/2
        seq = np.cos(seq)
        return seq

    d = into_cosine(d)
    dataframe = dataframe.replace(to_replace = np.NaN, value = 0)
    dataframe.loc[dataframe['cos'] == 1] = d
    return dataframe
def preprocess_data(data, is_train = True):
    a = pd.DataFrame()
    for i in range(int(len(data)/48)):
        tmp = pd.DataFrame()
        tmp['cos'] = data.loc[i*48:(i+1)*48-1,'TARGET']
        tmp['cos'] = make_cos(tmp)
        a = pd.concat([a,tmp])
    data['cos'] = a
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.insert(1,'Time',data['Hour']*2+data['Minute']/30.)
    temp = data.copy()
    temp = temp[['Time','TARGET','GHI','DHI','DNI','WS','RH','T']]

    if is_train == True:
        temp['TARGET1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['TARGET2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train == False:
        scale.transform
        return temp.iloc[-48*day:, :]

df_train = preprocess_data(train)
# scale.fit_transform(df_train.iloc[:,:-2])

df_test = []
for i in range(81):
    file_path = '../data/csv/test/%d.csv'%i
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp,is_train=False)
    temp = split_to_seq(temp)
    df_test.append(temp)

test = np.array(df_test)
train = split_to_seq(df_train)
# print(train.shape) #(48, 1093, 10)
# print(test.shape) #(81, 48, 4, 8)

def split_xy(data,timestep):
    x, y1, y2 = [],[],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:-2]
        tmp_y1 = data[x_end-1:x_end,-2]
        tmp_y2 = data[x_end-1:x_end,-1]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return(np.array(x),np.array(y1),np.array(y2))

x,y1,y2 = [],[],[]
for i in range(48):
    tmp1,tmp2,tmp3 = split_xy(train[i],day)
    x.append(tmp1)
    y1.append(tmp2)
    y2.append(tmp3)

x = np.array(x)
y1 = np.array(y1)
y2 = np.array(y2)
# print(x.shape, test.shape, y1.shape, y2.shape) (48, 1090, 4, 8) (81, 48, 4, 8) (48, 1090, 1) (48, 1090, 1)

# def split_x(data,timestep):
#     x = []
#     for i in range(len(data)):
#         x_end = i + timestep
#         if x_end>len(data):
#             break
#         tmp_x = data[i:x_end]
#         x.append(tmp_x)
#     return(np.array(x))

# x_test = split_x(x_test,1)
# # y1 을 내일의 타겟, y2 를 모레의 타겟!!

from sklearn.model_selection import train_test_split as tts

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)
quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

#2. 모델링
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, factor = 0.25, verbose = 1)
epochs = 1000
bs = 32


for i in range(48):
    print(f'{int(i/2)}시 {i%2*30}분 시간대 진행중...')
    # x_train, x_val, y1_train, y1_val, y2_train, y2_val = tts(x[i],y1[i],y2[i], train_size = 0.7,shuffle = True, random_state = 0)
    # 내일!
    for j in quantiles:
        filepath_cp = f'../data/modelcheckpoint/dacon_{i:2d}_y1seq_{j:.1f}.hdf5'
        model = load_model(filepath_cp, compile = False)
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        x = []
        for k in range(81):
            x.append(test[k,i])
        x = np.array(x)
        df_temp1 = pd.DataFrame(model.predict(x).round(2))
        # df_temp1 = pd.concat(pred, axis = 0)
        df_temp1[df_temp1<0] = 0
        num_temp1 = df_temp1.to_numpy()
        if i%2 == 0:
            submission.loc[submission.id.str.contains(f"Day7_{int(i/2)}h00m"), [f"q_{j:.1f}"]] = num_temp1
        elif i%2 == 1:
            submission.loc[submission.id.str.contains(f"Day7_{int(i/2)}h30m"), [f"q_{j:.1f}"]] = num_temp1

    # 모레!
    for j in quantiles:
        filepath_cp = f'../data/modelcheckpoint/dacon_{i:2d}_y2seq_{j:.1f}.hdf5'
        model = load_model(filepath_cp, compile = False)
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        x = []
        for k in range(81):
            x.append(test[k,i])
        x = np.array(x)
        df_temp2 = pd.DataFrame(model.predict(x).round(2))
        # df_temp1 = pd.concat(pred, axis = 0)
        df_temp2[df_temp2<0] = 0
        num_temp2 = df_temp2.to_numpy()
        if i%2 == 0:
            submission.loc[submission.id.str.contains(f"Day8_{int(i/2)}h00m"), [f"q_{j:.1f}"]] = num_temp2
        elif i%2 == 1:
            submission.loc[submission.id.str.contains(f"Day8_{int(i/2)}h30m"), [f"q_{j:.1f}"]] = num_temp2

submission.to_csv('../data/csv/0122_timeseries_noscale.csv', index = False)