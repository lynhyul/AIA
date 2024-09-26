# 탐색이란 많은 양의 데이터 중에서 원하는 데이터를 찾는 과정을 말한다.
# 대표적인 그래프 탐색 알고리즘으로는 DFS BFS가 있다.
# DFS BFS는 코딩테스트에서 매우 자주 등장하는 유형이므로 반드시 숙지!!!

# 스택 자료 구조
# 먼저 들어온 데이터가 나중에 나가는 형식의 자료구조
# 입구와 출구가 동일한 형태로 스택을 시각화 할 수 있다. 박스쌓기 예시로 기억 할 것

# 스택 동작 예시
# 삽입(5)-삽입(2)-삽입(3)-삽입(7)-삭제()-삽입(1)-삽입(4)-삭제()
s = []
s.append(5); s.append(2); s.append(3); s.append(7); s.pop(); s.append(1); s.append(4); s.pop()
print(s[::-1])  # [1, 3, 2, 5] 최상단
print(s)    # [5, 2, 3, 1] 최하단

# 큐 자료구조
# 먼저 들어 온 데이터가 먼저 나가는 형식(선입선출)의 자료구조
# 큐는 입구와 출구가 모두 뚫려있는 터널과 같은 형태로 시각화 할 수 있다. (은행창구에서의 번호표)

# 큐 동작 예시
# 삽입(5)-삽입(2)-삽입(3)-삽입(7)-삭제()-삽입(1)-삽입(4)-삭제()
from collections import deque
q = deque()
q.append(5); q.append(2); q.append(3); q.append(7); q.popleft(); q.append(1); q.append(4); q.popleft()

print(q)    # deque([3, 7, 1, 4])
q.reverse()
print(q)    # deque([4, 1, 7, 3])

# 재귀함수
# 자기자신을 다시 호출하는 함수를 의미
def recursive_function(i):
    if i == 100:
        return
    print(i,"번째 재귀함수에서", i+1,"번째 재귀함수를 호출합니다.")
    recursive_function(i+1)
    print(i, '번째 재귀함수를 종료합니다.')

recursive_function(90)  


# 최대공약수를 구하는 방법 / 유클리드 호재법

from math import gcd
g = gcd(192,162)    # 6
print(g)

# DFS
# 깊이 우선 탐색이라고도 부르며 그래프에서 깊은 부분을 우선적으로 탐색하는 알고리즘
# 과정
# 1. 탐색 시작 노드를 스택에 삽입하고 방문 처리를 한다
# 2. 스택의 최상단 노드에 방문하지 않은 인접한 노드가 하나라도 있으면 그 노드를 스택에 넣고 방문
# 처리를 한다 방문하지 않은 인접 노드가 없으면 스택에서 최상단 노드를 꺼낸다
# 3. 더 이상 2번의 과정을 수행 할 수 없을 때까지 반복한다

#def 매서드 정의
def dfs(graph, v, visited) :
    # 현재 노드를 방문 처리
    visited[v] = True
    # 현재 노드와 연결된 다른 노드를 재귀적으로 방문 
    print(v, end = ' ')
    for i in graph[v] :
        if not visited[i]:
            dfs(graph, i, visited)

graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]

# 각 노드가 방문된 정보를 표현 (1차원 리스트)
visited = [False] * 9

#정의된 DFS 함수 호출
print(dfs(graph, 1, visited))  # 1 2 7 6 8 3 4 5


# BFS
# 너비 우선탐색이라고도 부르며, 그래프에서 가까운 노드부터 우선적으로 탐색하는 알고리즘
# 큐 자료구조 이용
# 과정
# 1. 탐색 시작 노드를 큐에 삽입하고 방문 처리
# 2. 큐에서 노드를 꺼낸 뒤에 해당 노드의 인접 노드 중에서 방문하지 않는 노드를 모두 큐에 삽입하고 방문처리
# 3. 더 이상 2번의 과정을 수행할 수 없을 때까지 반복한다.

def bfs(graph, start, visited) :
    # 현재 노드를 방문 처리
    queue = deque([start])
    visited[start] = True
    # 큐가 빌 때까지 반복
    while queue :
        v = queue.popleft()
    # 현재 노드와 연결된 다른 노드를 재귀적으로 방문 
        print(v, end = ' ')
        for i in graph[v] :
            if not visited[i]:
                queue.append(i)
                visited[i] = True

graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]

visited = [False] * 9

print(bfs(graph, 1, visited))   # 1 2 3 8 7 4 5 6

