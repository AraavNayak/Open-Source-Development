import sys

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    ptr = 0
    T = int(input_data[ptr])
    k = int(input_data[ptr+1])
    ptr += 2
    
    for _ in range(T):
        FjString = int(input_data[ptr])
        S = input_data[ptr+1]
        ptr += 2
        
        n = 3 * FjString
        
        if n % 2 != 0:
            print("-1")
            continue
        
        if S[:n//2] == S[n//2:]:
            print(1)
            print(*([1] * n))
            continue
            
        c_pos = [i for i, char in enumerate(S) if char == 'C']
        o_pos = [i for i, char in enumerate(S) if char == 'O']
        w_pos = [i for i, char in enumerate(S) if char == 'W']
        
        if k == 1:
            print(n // 2)
            res = [0] * n
            op = 1
            for pos_list in [c_pos, o_pos, w_pos]:
                for i in range(0, len(pos_list), 2):
                    res[pos_list[i]] = op
                    res[pos_list[i+1]] = op
                    op += 1
            print(*res)
        else:
            mid = n // 2
            print(2)
            res = [0] * n
            c_half, o_half, w_half = FjString // 2, FjString // 2, FjString // 2
            
            c_count, o_count, w_count = 0, 0, 0
            for i, char in enumerate(S):
                if char == 'C':
                    c_count += 1
                    res[i] = 1 if c_count <= (c_half // 2) * 2 else 2
                elif char == 'O':
                    o_count += 1
                    res[i] = 1 if o_count <= (o_half // 2) * 2 else 2
                else:
                    w_count += 1
                    res[i] = 1 if w_count <= (w_half // 2) * 2 else 2
            
            op1_chars = []
            for i in range(n):
                if res[i] == 1: op1_chars.append(S[i])
            
            s_op1 = "".join(op1_chars)
            if s_op1[:len(s_op1)//2] != s_op1[len(s_op1)//2:]:
                for i in range(n):
                    res[i] = 1 if i < n//2 else 2
            print(*res)

solve()