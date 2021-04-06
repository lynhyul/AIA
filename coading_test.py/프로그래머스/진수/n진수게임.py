# 입력 형식
# 진법 n, 미리 구할 숫자의 갯수 t, 게임에 참가하는 인원 m, 튜브의 순서 p 가 주어진다.

# 2 ≦ n ≦ 16
# 0 ＜ t ≦ 1000
# 2 ≦ m ≦ 100
# 1 ≦ p ≦ m
# 출력 형식
# 튜브가 말해야 하는 숫자 t개를 공백 없이 차례대로 나타낸 문자열. 단, 10~15는 각각 대문자 A~F로 출력한다.

# 입출력 예제
# n	t	m	p	result
# 2	4	2	1	"0111"
# 16	16	2	1	"02468ACE11111111"
# 16	16	2	2	"13579BDF01234567"



def solution(n,t, m,p):
    
    def convert(n, base) :
        arr = "0123456789ABCDEF"
        q, r = divmod(n, base) # 몫과 나머지를 반환받는 함수. q = 몫 / r = 나머지
        if q == 0:
            return arr[r]
        else :
            return convert(q,base) + arr[r]
    answer = ''
    candidate = []

    for i in range(t*m) :
        conv = convert(i,n)
        for c in conv:
            candidate.append(c)
    for i in range(p-1, t*m, m):
        answer += candidate[i]
    return answer
    
print(solution(10,10,1,1))    # 01110