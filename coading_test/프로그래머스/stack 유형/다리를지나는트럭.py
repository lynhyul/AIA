# 문제 출저 : https://programmers.co.kr/learn/courses/30/lessons/42583

# 트럭 여러 대가 강을 가로지르는 일 차선 다리를 정해진 순으로 건너려 합니다. 
# 모든 트럭이 다리를 건너려면 최소 몇 초가 걸리는지 알아내야 합니다. 트럭은 1초에 1만큼 움직이며, 
# 다리 길이는 bridge_length이고 다리는 무게 weight까지 견딥니다.
# ※ 트럭이 다리에 완전히 오르지 않은 경우, 이 트럭의 무게는 고려하지 않습니다.

# 예를 들어, 길이가 2이고 10kg 무게를 견디는 다리가 있습니다. 무게가 [7, 4, 5, 6]kg인 트럭이 
# 순서대로 최단 시간 안에 다리를 건너려면 다음과 같이 건너야 합니다.

# bridge_length	    weight	    truck_weights	                  return
# 2	                 10	          [7,4,5,6]	                        8
# 100	             100           	[10]	                       101
# 100		         100     [10,10,10,10,10,10,10,10,10,10]       110



bridge_length = 2
weight = 10
truck_weights = [7,4,5,6]

from collections import deque

def solution(bridge_length, weight, truck_weights):
    bridge = deque(0 for _ in range(bridge_length))
    print(bridge)   # deque([0,0])
    total_weight = 0    # 현재 다리위에 달리고 있는 트럭의 총 무게
    step = 0            # 시간 경과
    truck_weights.reverse()
    print(truck_weights)    # [6, 5, 4, 7] => pop 할때마다 reverse를하면 속도가 느려져서 리스트를 한 번만 리버스 해준것.

    while truck_weights:
        total_weight -= bridge.popleft()    # 처음에는 0
        print(total_weight) # 0 7 0 4 5 0
        if total_weight + truck_weights[-1] > weight:   # 다리 위 트럭에 새로 출발하려는 트럭의 무게의 합이 다리 weight보다 높다면
            bridge.append(0)        # 다리 위 트럭 목록에 0을 추가하겠다.
        else:
            truck = truck_weights.pop() # 출발한 트럭은 대기목록에 있던 트럭에서 빼겠다.
            bridge.append(truck)    # 출발한 트럭을 다리 위 트럭 목록에 추가하겠다.
            total_weight += truck   # 다리 위 총 무게에 출발한 트럭의 무게를 추가하겠다.
        step += 1           # 시간경과 변수인 step의 크기를 1 추가하겠다.

    step += bridge_length

    return step

solution(bridge_length, weight, truck_weights)