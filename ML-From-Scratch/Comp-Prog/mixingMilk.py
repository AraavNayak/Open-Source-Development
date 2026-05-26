import sys


def solve():

    read = open("mixmilk.in")

    firstLine = read.readline().split()
    secondLine = read.readline().split()
    thirdLine = read.readline().split()

    caps = [int(firstLine[0]), int(secondLine[0]), int(thirdLine[0])]
    amts = [int(firstLine[1]), int(secondLine[1]), int(thirdLine[1])]

    def pour(indexOne, indexTwo):
        orig = amts[indexTwo]
        amts[indexTwo] = amts[indexTwo] + amts[indexOne]
        if amts[indexTwo] > caps[indexTwo]:
            amts[indexTwo] =  caps[indexTwo]
        diff = amts[indexTwo] - orig
        amts[indexOne] -= diff

    for i in range(33):
        pour(0, 1)
        pour(1, 2)
        pour(2, 0)
    pour(0, 1)

    with open("mixmilk.out", "w") as out:
        print(amts[0], file=out)
        print(amts[1], file=out)
        print(amts[2], file=out)






if __name__ == '__main__':
    solve()