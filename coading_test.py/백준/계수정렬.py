# ============= 계수 정렬 =================

# 특정한 데이터에서만 사용이 가능하나 매우 빠른 정렬 알고리즘이다
# 보통 성적 데이터 같은 곳에 많이 쓰인다

array = [7,5,9,0,3,1,6,2,9,1,4,8,0,5,2]

# 모든 범위를 포함하는 리스트 선언(모든 값은 0으로 초기화)
count = [0] * (max(array) + 1)

for i in range(len(array)):
    count[array[i]] += 1 # 각 데이터에 해당하는 인덱스의 값 증가

for i in range(len(count)): # 리스트에 기록된 정렬 정보 확인
    for j in range(count[i]):
        print(i, end='') # 띄워쓰기를 구분으로 횟수만큼 인덱스 출력

# ================ 삽입정렬 ===============

# 데이터가 거의 정렬이 되었을때 쓰이는게 좋음
# 삽입정렬은 두 번째 데이터 부터 시작

array = [7,5,9,0,3,1,6,2,4,8]

for i in range(1, len(array)): # 인덳스 0(맨앞)은 고정 1부터 시작
    for j in range(i, 0, -1): # 인덱스 i부터 1까지 감소하여 반복하는 문법
        if array[j] < array[j-1]: # 한 칸씩 왼쪽으로 이동하는 코드 -1
            array[j], array[j-1] = array[j-1], array[j] # 스와프
        else: # 자기보다 작은 데이터를 만나면 그 위치에서 멈춤
          break
print(array)

# =========== 선택 정렬 =============
# 매번 작은 것을 선택한다는 의미에서 선택행렬이라고 한다

# 선택 정렬 소스 코드
array = [7,5,9,0,3,1,6,2,4,8]

for i in range(len(array)):
    min_index = i # 가장 작은 원소의 인덱스
    for j in range(i + 1, len(array)):
        if array[min_index] > array[j]:
            min_index = j
    array[i], array[min_index] = array[min_index], array[i] # 스와프

print(array)
# 스와프 : 특정한 리스트가 주어졌을 때 두 변수의 위치를 변경하는 작업을 의미한다
# 데이터가 큰것에는 안쓰고 작은것에 자주쓰인다
# 알고리듬 문제풀이로는 좀 느리다