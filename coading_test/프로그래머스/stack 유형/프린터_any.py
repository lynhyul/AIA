
# 문제 출저
# https://programmers.co.kr/learn/courses/30/lessons/42587?language=python3#

# 예를 들어, 4개의 문서(A, B, C, D)가 순서대로 인쇄 대기목록에 있고 중요도가 2 1 3 2 라면 C D A B 순으로 인쇄하게 됩니다.

# 내가 인쇄를 요청한 문서가 몇 번째로 인쇄되는지 알고 싶습니다. 위의 예에서 C는 1번째로, A는 3번째로 인쇄됩니다.

# 현재 대기목록에 있는 문서의 중요도가 순서대로 담긴 배열 priorities와 내가 인쇄를 요청한 문서가 현재 대기목록의 
# 어떤 위치에 있는지를 알려주는 location이 매개변수로 주어질 때, 내가 인쇄를 요청한 문서가 몇 번째로 인쇄되는지
#  return 하도록 solution 함수를 작성해주세요.

# 제한사항
# 현재 대기목록에는 1개 이상 100개 이하의 문서가 있습니다.
# 인쇄 작업의 중요도는 1~9로 표현하며 숫자가 클수록 중요하다는 뜻입니다.
# location은 0 이상 (현재 대기목록에 있는 작업 수 - 1) 이하의 값을 가지며 대기목록의 가장 앞에 있으면 0, 두 번째에 있으면 1로 표현합니다.
# 입출력 예
# priorities	location	return
# [2, 1, 3, 2]	2	1
# [1, 1, 9, 1, 1, 1]	0	5
# 입출력 예 설명
# 예제 #1

# 문제에 나온 예와 같습니다.

# 예제 #2

# 6개의 문서(A, B, C, D, E, F)가 인쇄 대기목록에 있고 중요도가 1 1 9 1 1 1 이므로 C D E F A B 순으로 인쇄합니다.

#===========================================================================================================================


priorities = [1, 1, 9, 1, 1, 1]
location = 1



def solution(priorities, location):
    queue =  [(i,p) for i,p in enumerate(priorities)]
    print(queue)    # [(0, 1), (1, 1), (2, 9), (3, 1), (4, 1), (5, 1)]
    answer = 0  
    while True:
        cur = queue.pop(0)
        print(cur)
        if any(cur[1] < q[1] for q in queue):   # cur의 key값, queue의 key값을 비교
        # any(iterable) 함수는 인자로 받은 반복가능한 자료형(iterable)중 단 하나라도 참(True)이 있으면 참(True)를 반환하는 함수
            queue.append(cur)
        else:
            answer += 1
            if cur[0] == location:
                return answer
print(solution(priorities,location))