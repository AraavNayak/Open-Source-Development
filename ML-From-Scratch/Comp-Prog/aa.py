import sys
input = sys.stdin.readline

def find_square_suffix(S):
    n = len(S)
    for l in range(n // 2, 0, -1):
        if S[-2*l:-l] == S[-l:]:
            return n - 2*l, n
    return None

T, k = map(int, input().split())
for _ in range(T):
    FjString = int(input())
    S = input().strip()
    if FjString % 2 == 1:
        print(-1)
        continue
    res = [0]*len(S)
    ops = 0
    s = S
    while s:
        suffix = find_square_suffix(s)
        if not suffix:
            # if no square suffix, remove first two chars (smallest square)
            ops += 1
            res[len(S)-len(s)] = ops
            res[len(S)-len(s)+1] = ops
            s = s[2:]
        else:
            start, end = suffix
            ops += 1
            for i in range(start, end):
                res[len(S)-len(s)+i] = ops
            s = s[:start]
    print(ops)
    print(" ".join(map(str,res)))
