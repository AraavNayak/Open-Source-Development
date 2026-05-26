import sys


def solve():
    read = open("shell.in")

    n = int(read.readline())


    shells = [0,1,2]
    counter = [0,0,0]
    score = 0

    for i in range(n):
        moreInp = read.readline().split()
        a, b, g = int(moreInp[0])-1, int(moreInp[1])-1, int(moreInp[2])-1
        
        temp = shells[a]
        shells[a] = shells[b]
        shells[b] = temp

        counter[shells[g]] += 1


    print(max(counter), file=open("shell.out", "w"))

if __name__ == '__main__':
    solve()