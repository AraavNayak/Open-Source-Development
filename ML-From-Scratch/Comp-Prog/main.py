import sys
input = sys.stdin.readline

def solve():
    T = int(input())
    
    for _ in range(T):
        A, B, cA, cB, fA = map(int, input().split())
        
        if A >= fA:
            print(0)
            continue
        
        def min_type_a_achievable(x):
         
            
            def calc_final_a(a):
                exchanges = (B + x - a) // cB
                return A + a + exchanges * cA
            
            candidates = set()
            candidates.add(0)
            candidates.add(x)
            
            target = (B + x) % cB
            for candidate_a in [target - 1, target, target + 1]:
                if 0 <= candidate_a <= x:
                    candidates.add(candidate_a)
            
           
            remainder = (B + x) % cB
            if remainder <= x:
                candidates.add(remainder)
            if remainder + cB <= x:
                candidates.add(remainder + cB)
            
            return min(calc_final_a(a) for a in candidates)
        
        # Binary search
        lo, hi = 0, 2 * 10**10
        while lo < hi:
            mid = (lo + hi) // 2
            if min_type_a_achievable(mid) >= fA:
                hi = mid
            else:
                lo = mid + 1
        
        print(lo)

if __name__ == "__main__":
    solve()