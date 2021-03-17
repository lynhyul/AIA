# 61카피해서
# model.cv_result를 붙여서 완성


import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()

#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.


#2. 모델
def build_model(drop=0.5, optimizer = 'adam', node = 1, layer_num = 1) :
    input = Input(shape=(x_train.shape[1]), name = 'input')
    x = Dense(512/node, activation='relu', kernel_initializer='normal', name= 'hidden1') (input)
    x = Dropout(drop)(x)
    for i in range(layer_num) :
        x = Dense(256/node, activation='relu', kernel_initializer='normal',name= f'hidden2_{i}') (x)
        x = Dropout(drop)(x)
    x = Dense(32, activation='relu',  kernel_initializer='normal',name= 'hidden3') (x)
    x = Dropout(drop)(x)
    output = Dense(10, activation= 'softmax' ,name = 'output') (x)
    model = Model(inputs = input, outputs = output)
    model.compile(optimizer = optimizer, metrics=['acc'],
                    loss = 'categorical_crossentropy')
    return model

def create_hyperparmeters() :
    # batches = [10, 20, 30, 40, 50]
    opitmizer = ['adam']
    dropout = [0.2,0.3]
    node = [1,2,4]
    layer_num = [2,5,6,7]
    return {"optimizer" : opitmizer,
            "drop": dropout, "node" : node, "layer_num" : layer_num}     


hyperparmeters = create_hyperparmeters()
# model2 = build_model()    타입오류



from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose = 1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# mc = ModelCheckpoint('../data/modelcheckpoint/hyper1.h5',save_best_only=True, verbose=1)
# es = EarlyStopping(monitor = 'val_loss',patience=5)
# lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3, factor=0.5)

search = GridSearchCV(model2,hyperparmeters,cv =5)

search.fit(x_train, y_train,verbose = 1, validation_split = 0.2, batch_size=128)


print(search.best_params_)  # 선택한 파라미터중에서 가장 좋은거
print(search.cv_results_)

# print(search.best_estimator_)   # 전체 파라미터 중에서 가장 좋은거
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001CA15E32C40>

print(search.best_score_)   # 밑에 있는 .score랑은 결과가 다르게 나온다.


acc = search.score(x_test,y_test)
print("최종 스코어 : ",acc)

'''
{'drop': 0.2, 'layer_num': 2, 'node': 1, 'optimizer': 'adam'}
{'mean_fit_time': array([1.36916494, 1.12109451, 1.0957787 , 1.45421042, 1.38710403,
       1.37195334, 1.56114297, 1.49685192, 1.46968579, 1.67181067,
       1.60085025, 1.59488235, 1.17346706, 1.1149941 , 1.1112453 ,
       1.45006714, 1.33079839, 1.38424783, 1.58797317, 1.48759265,
       1.46376424, 1.74609966, 1.59855566, 1.49740057]), 'std_fit_time': array([0.46049597, 0.06036274, 0.04992187, 0.04590385, 0.07838946,
       0.08035982, 0.07641366, 0.08158703, 0.08727647, 0.10339863,
       0.10438306, 0.12800649, 0.11637216, 0.11463198, 0.12066783,
       0.11290521, 0.01128508, 0.14163725, 0.14060585, 0.13707591,
       0.1440313 , 0.1735132 , 0.17068125, 0.02297354]), 'mean_score_time': array([0.5836729 , 0.57639098, 0.56354527, 0.61058025, 0.64440451,
       0.61532688, 0.63551688, 0.63828392, 0.63957758, 0.66046371,
       0.66669884, 0.65831895, 0.5729135 , 0.55791693, 0.56774282,
       0.61443176, 0.72663064, 0.61985788, 0.65877094, 0.64877052,
       0.64480262, 0.66274948, 0.65850167, 0.73858786]), 'std_score_time': array([0.03723362, 0.04121602, 0.00258671, 0.00303995, 0.06310163,
       0.00485906, 0.00419123, 0.01084401, 0.00574844, 0.01077107,
       0.02344892, 0.00258746, 0.02416394, 0.00486932, 0.00316124,
       0.00928842, 0.14738776, 0.01227246, 0.01318237, 0.01074516,
       0.00920229, 0.01182598, 0.00394395, 0.14896525]), 'param_drop': masked_array(data=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                   0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                   0.3, 0.3],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_layer_num': masked_array(data=[2, 2, 2, 5, 5, 5, 6, 6, 6, 7, 7, 7, 2, 2, 2, 5, 5, 5,
                   6, 6, 6, 7, 7, 7],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_node': masked_array(data=[1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4,
                   1, 2, 4, 1, 2, 4],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_optimizer': masked_array(data=['adam', 'adam', 'adam', 'adam', 'adam', 'adam', 'adam',
                   'adam', 'adam', 'adam', 'adam', 'adam', 'adam', 'adam',
                   'adam', 'adam', 'adam', 'adam', 'adam', 'adam', 'adam',
                   'adam', 'adam', 'adam'],
             mask=[False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'params': [{'drop': 0.2, 'layer_num': 2, 'node': 1, 'optimizer': 'adam'}, {'drop': 0.2, 'layer_num': 2, 'node': 2, 'optimizer': 'adam'}, {'drop': 0.2, 'layer_num': 2, 'node': 4, 'optimizer': 'adam'}, {'drop': 0.2, 'layer_num': 5, 'node': 1, 'optimizer': 'adam'}, {'drop': 0.2, 'layer_num': 5, 'node': 2, 'optimizer': 'adam'}, {'drop': 0.2, 'layer_num': 5, 'node': 4, 'optimizer': 'adam'}, {'drop': 0.2, 'layer_num': 6, 'node': 1, 'optimizer': 'adam'}, {'drop': 0.2, 'layer_num': 6, 'node': 2, 'optimizer': 'adam'}, {'drop': 0.2, 'layer_num': 6, 'node': 4, 'optimizer': 'adam'}, {'drop': 0.2, 'layer_num': 7, 'node': 1, 'optimizer': 'adam'}, {'drop': 0.2, 'layer_num': 7, 'node': 2, 'optimizer': 'adam'}, {'drop': 0.2, 'layer_num': 7, 'node': 4, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 2, 'node': 1, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 2, 'node': 2, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 2, 'node': 4, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 5, 'node': 1, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 5, 
'node': 2, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 5, 'node': 4, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 6, 'node': 1, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 6, 'node': 2, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 6, 'node': 4, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 7, 'node': 1, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 7, 'node': 2, 'optimizer': 'adam'}, {'drop': 0.3, 'layer_num': 7, 'node': 4, 'optimizer': 'adam'}], 'split0_test_score': array([0.95450002, 0.94433331, 0.91850001, 0.95108336, 0.92075002,
       0.78583336, 0.94008332, 0.84083331, 0.77216667, 0.92816669,
       0.7759167 , 0.65833336, 0.94741666, 0.94066668, 0.91074997,
       0.94524997, 0.92624998, 0.79083335, 0.93016666, 0.75716668,
       0.69041669, 0.83458334, 0.69091666, 0.26858333]), 'split1_test_score': array([0.95408332, 0.93808335, 0.92025   , 0.94583333, 0.92725003,
       0.77933335, 0.94258332, 0.87258333, 0.57716668, 0.93774998,
       0.85841668, 0.551     , 0.94625002, 0.93583333, 0.91000003,
       0.93841666, 0.90058333, 0.54033333, 0.91891664, 0.87991667,
       0.37391666, 0.91566664, 0.6566667 , 0.37658334]), 'split2_test_score': array([0.94591665, 0.94191664, 0.90733331, 0.94524997, 0.92408335,
       0.71591669, 0.94041669, 0.74425   , 0.62625003, 0.93483335,
       0.76558334, 0.57975   , 0.94450003, 0.93141669, 0.89749998,
       0.93441665, 0.88849998, 0.67449999, 0.89108336, 0.81766665,
       0.4725    , 0.92075002, 0.5735833 , 0.3545    ]), 'split3_test_score': array([0.95025003, 0.93416667, 0.91100001, 0.93774998, 0.91083336,
       0.88616669, 0.94199997, 0.91958332, 0.67358333, 0.92708331,
       0.8326667 , 0.62674999, 0.94408333, 0.93233335, 0.90266669,
       0.93349999, 0.83999997, 0.55425   , 0.93591666, 0.73474997,
       0.53383332, 0.829     , 0.71891665, 0.28716666]), 'split4_test_score': array([0.95508331, 0.94483334, 0.91808331, 0.93366665, 0.92650002,
       0.8175    , 0.94733334, 0.92250001, 0.66316664, 0.94050002,
       0.75683331, 0.45216668, 0.94983333, 0.94158334, 0.92299998,
       0.93650001, 0.90966666, 0.74416667, 0.93766665, 0.88116664,
       0.48908332, 0.91233331, 0.7015    , 0.51958334]), 'mean_test_score': array([0.95196667, 0.94066666, 0.91503333, 0.94271666, 0.92188336,
       0.79695002, 0.94248333, 0.85994999, 0.66246667, 0.93366667,
       0.79788334, 0.57360001, 0.94641668, 0.93636668, 0.90878333,
       0.93761666, 0.89299998, 0.66081667, 0.92275   , 0.81413332,
       0.51195   , 0.88246666, 0.66831666, 0.36128333]), 'std_test_score': array([0.00346875, 0.00403216, 0.0049818 , 0.00620812, 0.0059723 ,
       0.05546574, 0.00259989, 0.0653914 , 0.06441412, 0.00525965,
       0.04021217, 0.07113269, 0.00208899, 0.00416553, 0.00862773,
       0.00417951, 0.02922362, 0.09991144, 0.01713677, 0.06063054,
       0.10340371, 0.04150028, 0.05154102, 0.08879816]), 'rank_test_score': array([ 1,  5, 11,  3, 10, 18,  4, 15, 20,  8, 17, 22,  2,  7, 12,  6, 13,
       21,  9, 16, 23, 14, 19, 24])}
0.951966667175293
  1/313 [..............................] - ETA: 0s - loss: 0.0772 - acc: 0.9688WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0000s vs `on_test_batch_end` time: 0.0156s). Check your callbacks.
313/313 [==============================] - 0s 2ms/step - loss: 0.1606 - acc: 0.9531
최종 스코어 :  0.9531000256538391
'''