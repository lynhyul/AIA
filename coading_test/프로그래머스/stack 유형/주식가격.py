# 문제 링크 : https://programmers.co.kr/learn/courses/30/lessons/42584

# 문제 설명
# 초 단위로 기록된 주식가격이 담긴 배열 prices가 매개변수로 주어질 때, 가격이 떨어지지 않은 기간은 
# 몇 초인지를 return 하도록 solution 함수를 완성하세요.

# 제한사항
# prices의 각 가격은 1 이상 10,000 이하인 자연수입니다.
# prices의 길이는 2 이상 100,000 이하입니다.
# 입출력 예
# prices	return
# [1, 2, 3, 2, 3]	[4, 3, 1, 1, 0]
# 입출력 예 설명
# 1초 시점의 ₩1은 끝까지 가격이 떨어지지 않았습니다.
# 2초 시점의 ₩2은 끝까지 가격이 떨어지지 않았습니다.
# 3초 시점의 ₩3은 1초뒤에 가격이 떨어집니다. 따라서 1초간 가격이 떨어지지 않은 것으로 봅니다.
# 4초 시점의 ₩2은 1초간 가격이 떨어지지 않았습니다.
# 5초 시점의 ₩3은 0초간 가격이 떨어지지 않았습니다.

#=======================================================================================================

# 데크(deque)의 개념
# 보통 큐(queue)는 선입선출(FIFO) 방식으로 작동한다. 반면, 양방향 큐가 있는데 그것이 바로 데크(deque)다.

# 즉, 앞, 뒤 양쪽 방향에서 엘리먼트(element)를 추가하거나 제거할 수 있다.

# 데크는 양 끝 엘리먼트의 append와 pop이 압도적으로 빠르다.

# 컨테이너(container)의 양끝 엘리먼트(element)에 접근하여 삽입 또는 제거를 할 경우, 일반적인 리스트(list)가 이러한 연산에 
# O(n)이 소요되는 데 반해, 데크(deque)는 O(1)로 접근 가능하다.

prices = [1, 2, 3, 2, 3]

from collections import deque


print(deque(prices))    # deque([1, 2, 3, 2, 3])
price = deque(prices)
print(price.popleft())  # 1. 가장 왼쪽의 요소를 pop 한다.

def solution(prices):
    answer = []
    prices = deque(prices)
    while prices:
        c = prices.popleft()
        # print(c)    # 1 2 3 2 3
        count = 0
        for i in prices:
            if c > i:   # pop으로 인해 나온 요소(맨왼쪽의 요소부터 시작 : 1)가 더 클 경우에는 빠져나간다.
                count += 1
                break
            count += 1  # 가격이 떨어지지 않는다면 카운트의 갯수를 쌓는다.

        answer.append(count)

    return answer


print(solution(prices)) # [4, 3, 1, 1, 0]