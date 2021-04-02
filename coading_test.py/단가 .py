# 15kg을 넣을수 있는 배낭에 물건들을 넣어 최고의 단가를 올려보자
# 4달러짜리 12kg짐, 10달러짜리 4kg짐, 2달러짜리 1kg짐, 2달러짜리 2kg짐, 1달러짜리 1kg짐
# 짐을 쪼갤 수 있다는 조건

cargo = [[4,12],[2,1],[10,4],[1,1],[2,2]]

def fractional_knapsack(cargo):
    capacity=15
    pack = []
    # 단가 계산 역순 정렬
    for c in cargo:
        pack.append([c[0]/c[1],c[0],c[1]]) # 가성비를 추가
    pack.sort(reverse=True) # 가성비가 좋은 순서대로 먼저 넣어줌

    # 단가 순 그리디 계산
    total_value: float = 0
    for p in pack:
        if capacity - p[2] >= 0: # 무게 초과 여부
            capacity -= p[2] # 무게만큼 가능용량에서 빼줌
            total_value += p[1] # 금액 계산
        else:
            # 물건 분할하여 채우기
            fraction = capacity / p[2] 
            total_value += p[1] * fraction
            break
    return total_value

print(fractional_knapsack(cargo))