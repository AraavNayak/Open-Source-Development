import sys

input = sys.stdin.readline

N, K = map(int, input().split())
Q = int(input())

beauty = [[0] * N for _ in range(N)]

M = N - K + 1

sums = [[0] * M for _ in range(M)]

current_max = 0

for _ in range(Q):
    r, c, v = map(int, input().split())
    r -= 1
    c -= 1

    delta = v - beauty[r][c]
    beauty[r][c] = v

    r_start = max(0, r - K + 1)
    r_end = min(r, M - 1)
    c_start = max(0, c - K + 1)
    c_end = min(c, M - 1)

    for i in range(r_start, r_end + 1):
        row_sums = sums[i]
        for j in range(c_start, c_end + 1):
            row_sums[j] += delta
            if row_sums[j] > current_max:
                current_max = row_sums[j]

    print(current_max)
