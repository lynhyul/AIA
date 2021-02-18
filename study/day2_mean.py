# #리스트 평균 값 구하기

# arr = [1,2,3,4]

# def solution(arr):
#     answer = sum(arr)/len(arr)
#     return answer
# print(solution(arr))

for i in range(10) :
    images = []
    for j in range(1,350) :
        try :
            print(f'../data/image/project/{i}/{i} ({j}).jpg')
        except :
            break
