# 정렬 이란 데이터를 특정한 기준에 따라 순서대로 나열하는 것을 말합니다.
# 일반적으로 문제 상황에 따라서 적절한 정렬 알고리즘이 공식처럼 사용됩니다.


# 선택 정렬
# 처리되지 않은 데이터 중에서 가장 작은 데이터를 선택해 맨 앞에 있는 데이터와 바꾸는 것을 반복.

array = [7,5,9,0,3,1,6,2,4,8]

for i in range(len(array)) :
    min_index = i   # 가장 작은 원소
    for j in range(i+1,len(array)) :
        if array[min_index] > array[j] :
            min_index = j
    array[i], array[min_index] = array[min_index], array[i] # 스와프
print(array)    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 선택 정렬의 시간 복잡도
# (N^2+N-2) /2 로 표현할 수 있는데, 빅오 표기법에 따라서 O(N^2)이라고 한다


# 삽입 정렬
# 처리되지 않은 데이터를 하나씩 골라 적절한 위치에 삽입한다
# 선택 정렬에 비해 구현 난이도가 높지만, 일반적으로 더 효율적으로 동작한다

array2 = [7,5,9,0,3,1,6,2,4,8]

for i in range(1, len(array2)) :
    for j in range(i, 0, -1) :  # 인덱스 i부터 1까지 1씩 감소하며 반복
        if array2[j] < array2[j-1] :  # 한 칸씩 왼쪽으로 이동
            array2[j], array2[j-1] = array2[j-1], array2[j]
        else :  # 자기보다 작은 데이터를 만나면 그 위치에서 멈춤
            break

print(array2)    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 삽입 정렬의 시간 복잡도
# 삽입 정렬의 시간 복잡도는 O(N^2)이며, 선택 정렬과 마찬가지로 반복문이 두 번 중첩되어 사용된다
# 삽입 정렬은 현재 리스트의 데이터가 거의 정렬되어 있는 상태라면 매우 빠르게 동작
# 최선의 경우 O(N)의 시간 복잡도를 가진다


# 퀵 정렬
# 기준 데이터를 설정하고 그 기준보다 큰 데이터와 작은 데이터의 위치를 바꾸는 방법
# 일반적인 상황에서 가장 많이 사용되는 정렬 알고리즘 중 하나
# 병합 정렬과 더불어 대부분의 프로그래밍 언어의 정렬 라이브러리의 근간이 되는 알고리즘
# 가장 기본적인 퀵 정렬은 첫 번째 데이터를 기준 데이터로 설정
# 퀵 정렬이 가장 빠른 이유는 이상적인 경우 분할이 절반씩 일어난다면 전체 연산 횟수로 O(NlogN)를
# 기대 할 수 있다 (너비 x 높이 = N x logN = NlogN)

# 퀵정렬의 시간 복잡도
# 퀵정렬은 평균의 경우 O(NlogN)의 시간 복잡도를 가진다
# 하지만 최악의 경우 O(N^2)의 시간 복잡도를 가진다


array3 = [5,7,9,0,3,1,6,2,4,8]

def quick_sort(array, start, end) :
    if start >= end :   # 원소가 1개인 경우
        return
    pivot = start   # 피벗은 첫 번째 원소
    left = start +1
    right = end
    while(left <= right) :
        # 피벗보다 큰 데이터를 찾을 때까지 반복
        while(left <= end and array[left] <= array[pivot]):
            left +=1
        # 피벗보다 작은 데이터를 찾을 때 까지 반복
        while(right > start and array[right] >= array[pivot]):
            right -=1
        if(left > right):   # 엇갈렸다면 작은 데이터와 피벗을 교체
            array[right], array[pivot] = array[pivot], array[right]
        else:   # 엇갈리지 않았다면 작은 데이터와 큰 데이터를 교체
            array[left], array[right] = array[right], array[left]
    # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행
    quick_sort(array, start, right-1)
    quick_sort(array, right +1 , end)
    
quick_sort(array3, 0, len(array) - 1)
print(array3)   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 파이썬의 장점을 살린 방식

array4 = [5,7,9,0,3,1,6,2,4,8]

def quick_sort2(array) :
    # 리스트가 하나 이하의 원소만을 담고 있다면 종료
    if len(array) <= 1:
        return array
    pivot = array[0]    # 피벗은 첫 번째 원소
    tail = array[1:]    # 피벗을 제외한 리스트

    left_side = [x for x in tail if x <= pivot] # 분할된 왼쪽 부분
    right_side = [x for x in tail if x > pivot] # 분할된 오른쪽 부분

    # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행하고, 전체 리스트 반환
    return quick_sort2(left_side) + [pivot] + quick_sort2(right_side)

print(quick_sort2(array4))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 계수 정렬
# 특정한 조건이 부합할 때만 사용할 수 있지만 매우 빠르게 동작하는 정렬 알고리즘
# 결과를 확인할 때는 리스트의 첫 번째 데이터부터 하나씩 그 값만큼 반복하여 인덱스를 출력

# 모든 원소의 값이 0보다 크거나 같다고 가정
arr = [7,5,9,0,3,1,6,2,9,1,4,8,0,5,2]
# 모든 범위를 포함하는 리스트 선언(모든 값은 0으로 초기화)
count = [0] * (max(arr) + 1)

for i in range(len(array)) :
    count[arr[i]] += 1 # 각 데이터에 해당하는 인덱스의 값 증가

for i in range(len(count)) :    # 리스트에 기록된 정렬 정보 확인
    for j in range(count[i]) :
        print(i, end = '')  # 띄어쓰기를 구분으로 등장한 횟수만큼 인덱스 출력