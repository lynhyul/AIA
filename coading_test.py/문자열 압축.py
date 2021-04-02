# 입출력 예
#   s          result
# "aabbaccc"   7
# "ababcdcdababcdcd"   9
# "abcabcdede"   8
# "abcabcabcabcdededededede"   14
# "xababcdcdababcdcd"   17

# 입출력 예에 대한 설명
# 입출력 예 #1

# 문자열을 1개 단위로 잘라 압축했을 때 가장 짧습니다.

# 입출력 예 #2

# 문자열을 8개 단위로 잘라 압축했을 때 가장 짧습니다.

# 입출력 예 #3

# 문자열을 3개 단위로 잘라 압축했을 때 가장 짧습니다.

# 입출력 예 #4

# 문자열을 2개 단위로 자르면 "abcabcabcabc6de" 가 됩니다.
# 문자열을 3개 단위로 자르면 "4abcdededededede" 가 됩니다.
# 문자열을 4개 단위로 자르면 "abcabcabcabc3dede" 가 됩니다.
# 문자열을 6개 단위로 자를 경우 "2abcabc2dedede"가 되며, 이때의 길이가 14로 가장 짧습니다.

# 입출력 예 #5

# 문자열은 제일 앞부터 정해진 길이만큼 잘라야 합니다.
# 따라서 주어진 문자열을 x / ababcdcd / ababcdcd 로 자르는 것은 불가능 합니다.
# 이 경우 어떻게 문자열을 잘라도 압축되지 않으므로 가장 짧은 길이는 17이 됩니다.

## =================================================================================
# 문제 해설
# 1. 문자열을 정해진 길이로 판별하며 반복해야 하고, 문자열의 반을 넘어가면 같은 문자열이 
# 나올 수 없으므로, 문자열의 절반 까지만 반복

# 2. 정해진 길이의 문자열 S가 다음 문자열과 같으면, 숫자를 올리고 다음 문자열과 다시 비교, 
# 다르면 이전의 문자열과 숫자를 합해 한곳에 저장하고, 문자열 S를 달라진 문자열로 변경하고 비교를 반복

# 3. 끝까지 비교하면 그 문자열의 길이를 배열 Length에 저장

# 4. 배열 Length의 최소값을 리턴
##====================================================================================


def compress(text, tok_len):
    words = [text[i:i+tok_len] for i in range(0, len(text), tok_len)]
    res = []
    cur_word = words[0]
    cur_cnt = 1
    for a, b in zip(words, words[1:] + ['']):
        if a == b:
            cur_cnt += 1
        else:
            res.append([cur_word, cur_cnt])
            cur_word = b
            cur_cnt = 1
    return sum(len(word) + (len(str(cnt)) if cnt > 1 else 0) for word, cnt in res)

def solution(text):
    return min(compress(text, tok_len) for tok_len in list(range(1, int(len(text)/2) + 1)) + [len(text)])

a = [
    "aabbaccc",
    "ababcdcdababcdcd",
    "abcabcdede",
    "abcabcabcabcdededededede",
    "xababcdcdababcdcd",
    'aaaaaa',
]

for x in a:
    print(solution(x))