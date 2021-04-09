# N * M 크기의 얼름 틀이 있다
# 구멍이 뚫려 있는 부분은 0, 칸막이 존재하는 부분은 1로 표시
# 상하좌우로 연결 돼 있으므로 그래프 형식으로 나타낼 수 있음

# 특정한 지점의 주변 상하좌우를 살펴본 뒤에 주변 지점 중에서 값이 

# N, M을 공백으로 구분하여 입력받기
n,m = map(int, input().split())

# 2차원 리스트의 맵 정보 입력받기(외우기)
graph = []
for i in range(n):
    graph.append(list(map(int, input())))

# DFS로 특정한 노드를 방문한 뒤에 연결된 모든 노드들도 방문
def dfs(x, y):
    # 주어진 범위를 벗어나는 경우에는 즉시 종료
    if x <= -1 or x >= n or y <= -1 or y >= m:
        return False
    # 현재 노드에 방문하지 않았다면
    if graph[x][y] == 0:
        # 해당노드 방문처리
        graph[x][y] == 1
        # 상하좌우의 위치도 모두 재귀적으로 호출
        dfs(x, -1, y)
        dfs(x, y, -1)
        dfs(x, +1, y)
        dfs(x, y, +1)
        return True
    return False

# 모든 노드에 대해 음료수 채우기
result = 0
for i in range(n):
    for j in range(m):
        # 현재 위치에서 DFS 수행
        if dfs(i, j) == True:
            result += 1

print(result)