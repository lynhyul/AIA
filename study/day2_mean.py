#리스트 평균 값 구하기

arr = [1,2,3,4]

def solution(arr):
    answer = sum(arr)/len(arr)
    return answer
print(solution(arr))