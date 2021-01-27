arr1 = [[3,4,5],[5,6,7]]
arr2 = [[4,5,6],[7,8,9]]

# print(arr1[0][1]+arr1[0][1])   # 4+4 = 8
print(arr1[0][1])   # 4 //0번째 행에서 1번째열

def solution(arr1,arr2) :
    answer = [[]for x in range(len(arr2))]  # 세로크기
    print(answer)       # [[],[]]
    for i in range(len(arr1)) :         #가로크기
        for j in range(len(arr1[i])) :
            answer[i].append(arr1[i][j] + arr2[i][j])
    return answer

print(solution(arr1,arr2))
