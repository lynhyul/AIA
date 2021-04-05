# 문제 설명
# 행렬의 덧셈은 행과 열의 크기가 같은 두 행렬의 같은 행, 같은 열의 값을 서로 더한 결과가 됩니다. 
# 2개의 행렬 arr1과 arr2를 입력받아, 행렬 덧셈의 결과를 반환하는 함수, solution을 완성해주세요.

# 제한 조건
# 행렬 arr1, arr2의 행과 열의 길이는 500을 넘지 않습니다.
# 입출력 예
# arr1	arr2	return
# [[1,2],[2,3]]	[[3,4],[5,6]]	[[4,6],[7,9]]
# [[1],[2]]	[[3],[4]]

arr1 = [[3,4,5],[5,6,7]]
arr2 = [[4,5,6],[7,8,9]]

# print(arr1[0][1]+arr1[0][1])   # 4+4 = 8
print(arr1[0][1])   # 4 //0번째 행에서 1번째열

def solution(arr1,arr2) :
    answer = [[]for x in range(len(arr2))]  # 세로크기
    for i in range(len(arr1)) :         #가로크기
        for j in range(len(arr1[i])) :
            answer[i].append(arr1[i][j] + arr2[i][j])
    return answer

print(solution(arr1,arr2))


# 다른 사람 풀이

def sumMatrix(A,B):
    answer = [[c + d for c, d in zip(a, b)] for a, b in zip(A,B)]
    return answer

# 아래는 테스트로 출력해 보기 위한 코드입니다.
print(sumMatrix([[1,2], [2,3]], [[3,4],[5,6]]))