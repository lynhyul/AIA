arr1 = [[1,2],[2,3]]
arr2 = [[3,4],[5,6]]

def solution(arr1, arr2):
    answer = []
    arr3 = []
    for x in range(len(arr1)) :
        arr3 = []
        for y in range(len(arr2)) :
            arr3.append(arr1[x][y]+arr2[x][y])
        answer.append(arr3)
    return answer
print(solution(arr1,arr2))