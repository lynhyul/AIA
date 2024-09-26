'''
def recursive_function(i):
    # 100번째 출력했을 때 종료 되도록 종료 조건 명시
    if i == 100:
        return
    print(i, '번째 재귀함수에서', i+1, '번째 재귀함수를 호출합니다.')
    recursive_function(i+1)
    print(i, '번째 재귀함수를 종료합니다')
recursive_function(1)
'''
# ========= 2가지 방식으로 구현한 팩토리얼의 예제 ===============
# 반족적으로 구현한 n!
def factorial_iterative(n):
    result = 1
    # 1부터 n까지의 수 차례대로 곱하기
    for i in range(1, n+1):
        result *= i
    return result

# 재귀적으로 구현한 n!
def factorial_recusive(n):
    if n <= 1: # n이 1 이하인 경우를 1을 반환
       return 1
    # n! = n * (n-1)!를 그대로 코드로 작성하기
    return n * factorial_recusive(n -1) # n이 1보다 클때

# 각각의 방식으로 구현한 n 출력
print('반복적으로 구현:', factorial_iterative(5)) 
print('재귀적으로 구현:', factorial_recusive(5)) 