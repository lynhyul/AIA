# BFS '너비 우선 탐색', 가까운 노드부터 탐색하는 알고리즘
# 1. 탐색 시작 노드를 큐에 삽입 후 방문 처리
# 2. 큐에서 노드를 꺼내 해당 노드의 인접노드 중에서 방문하지 않은 노드를
# 모두 큐에 삽입하고 방문처리를 한다
# 3. 2번의 과정을 더 이상 수행할 수 없을 때까지 반복한다

from collections import deque

# BFS 메서드 정의
def bfs(graph, start, visited):
    # 큐(queue) 구현을 위해 deque 라이브러리 사용
    queue = deque([start])
    # 현재 노드를 방문 처리
    visited[start] = True
    # 큐가 빌 때까지 반복
    while queue:
        # 큐에서 하나의 원소를 뽑아 출력
        v = queue.popleft()
        print(v, end ='')
        # 해당원소와 연결된 아직 방문하지 않은 큐에 삽입
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
    
# 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원리스트)
graph = [
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]    
]

# 각노드가 방문한 정보를 리스트 자료형으로 표현(1차원리스트)
visited = [False] * 9

# 정의된 함수 bfs로 표현
bfs(graph, 1, visited)

# ================= 답: 12387456
