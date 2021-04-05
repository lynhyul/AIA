# NxN 행렬의 자물쇠와 MxM 행렬의 열쇠
# 열쇠를 회전, 이동시켜서 자물쇠의 빈공간을 채울 수 있으면 True, 아니면 False
# 3 <= M, N <= 20   N >= M
# 단, 돌기끼리 만나면 안됨


# 열쇠 90도 돌려주는 함수
def rotate_90(a): # a는 열쇠 행렬
    n = len(a) 
    m = len(a[0])
    result = [[0]*n for _ in range(m)] # 열쇠와 동일한 shape의 빈 행렬 만들기
    for i in range(n):
        for j in range(m):
            result[j][n-i-1] = a[i][j] # 돌리기
    return result

def check(new_lock):
    lock_length = len(new_lock) // 3 # 행렬 크기를 3배로 (그래야 열쇠 크기가 자물쇠 크기랑 같아도 안겹치게 할 수 있음)
    for i in range(lock_length, lock_length*2): 
        for j in range(lock_length, lock_length*2): # 3배크기 행렬의 가운데 부분을 의미
            if new_lock[i][j] != 1: # 모두 1이 아니면 (빈공간0 or 돌기만남2)
                return False # 열쇠랑 안맞는다
    return True # 맞는다

def solution(key, lock):
    n = len(lock) # 자물쇠 크기
    m = len(key) # 열쇠 크기
    
    new_lock = [[0]*(n*3) for _ in range(n*3)] # 행렬 3배
    for i in range(n):
        for j in range(n):
            new_lock[i+n][j+n] = lock[i][j] # 자물쇠를 3배행렬 가운데에 안착
    
    for rotation in range(4): # 4방향으로 돌리기
        key = rotate_90(key) # 처음부터 돌리고 시작
        for x in range(n*2):
            for y in range(n*2):
                for i in range(m):
                    for j in range(m): # 열쇠의 활동 반경을 (2n+m)x(2n+m)으로 설정 -> 사실상 전범위
                        new_lock[x + i][y + j] += key[i][j] # 열쇠행렬+자물쇠행렬
                if check(new_lock) == True: # 열쇠+자물쇠로 3배 행렬의 가운데 NxN부분이 모두 1이면 True
                    return True
                for i in range(m): # 열쇠부분 제거하고 다시 끼우기 위한 for문 (시도 후 남아있으면 안되니까!)
                    for i in range(m):
                        new_lock[x + i][y + j] -= key[i][j] # +되었던거 그냥 원상 복구 시켜줌
    return False # 모든 경우의 수를 다 해봐도 가운데 NxN부분이 모두 1이 아니면 False