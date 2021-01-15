import numpy as np

a = np.array(range(1,101))
size = 6                #size를 5로 정의하겠다.

def split_x(seq, size) :            #입력을 seq,size 받아들여서 아래와 같이 행동하겠다.
    aaa = []            # 빈 리스트 생성
    for i in range(len(seq)- size+1) :  # len(np.array(range(1,11)))은 리스트의 길이를
                                        # 의미하므로, 10이다. 10-5(size)+1 = 6
                                        # 즉, i값을 0부터6까지 아래 구문을 반복한다는 뜻.
        subset = seq[i :  (i+size)]     # subset= seq[i : (i+size)]
                                        # 해석하면 np.array(range(1,11))의 [i:(i+size)]
                                        # i부터 (i+size)-1까지의 리스트
        #aaa.append([item for item in subset])   # i가 반복될동안 subset에 있는 리스트를
                                                # 추가해내간다.
        aaa.append(subset)
    print(type(aaa))                    #aaa의 타입을 출력해라
    return np.array(aaa)                #aaa의 리스트를 출력해라

dataset = split_x(a,size)
x = dataset[:,:5]           # :(행),:(렬) => 슬라이싱
y = dataset[:,-1:]
print("====================")
print(dataset)
print(dataset.shape)            # (6,5)
print(x)
print(y)
'''
aaa.append([item for item in subset])
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]

aaa.append(subset)
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]

 '''
#결과를 예측해보자면, a = [1~11]까지를 seq에 대입을,size를 size에 대입하여
#a[:-1]값인 10이 될때까지 순차적으로 반복 하되, 열의 크기는 size에 의해 결정된다.
#자동으로 열의 크기가 정해지면 10이 될때까지 수행만하면 되니 행도 그에 따라 크기가
#결정되는 함수이다.