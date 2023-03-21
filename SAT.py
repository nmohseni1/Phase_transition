import numpy as np
from itertools import combinations
import exact_cover as ec
#from pysat.solvers import Glucose3, Minisat22
Seed=123
def SAT3(n, m):
    # generate instance of 3SAT
    # n is number of variables, m is number of clauses
    np.random.seed()
    A = np.zeros((n, m), dtype=int)
    for i in range(m):
        clause = np.random.choice(n, 3, replace=False)
        sign = np.random.choice([-1, 1], size=n, replace=True)
        col = [sign[k] if k in clause else 0 for k in range(n)]
        A[:, i] = col
    return A

def SAT2(n, m):
    # generate instance of 2SAT
    # n is number of variables, m is number of clauses
    np.random.seed()
    A = np.zeros((n, m), dtype=int)
    for i in range(m):
        clause = np.random.choice(n, 2, replace=False)
        sign = np.random.choice([-1, 1], size=n, replace=True)
        col = [sign[k] if k in clause else 0 for k in range(n)]
        A[:, i] = col
    return A




def P1in3SAT(n, m):
    # generate instance of positive 1-in-3 SAT
    # n is number of variables, m is number of clauses
    #np.random.seed()
    A = np.zeros((n, m), dtype=int)
    for i in range(m):
        clause = np.random.choice(n, 3, replace=False)
        col = [1 if k in clause else 0 for k in range(n)]
        A[:, i] = col
    return A

def P1in2SAT(n, m):
    # generate instance of positive 1-in-2 SAT
    # n is number of variables, m is number of clauses
    #np.random.seed(Seed)
    A = np.zeros((n, m), dtype=int)
    for i in range(m):
        clause = np.random.choice(n, 2, replace=False)
        col = [1 if k in clause else 0 for k in range(n)]
        A[:, i] = col
    return A

#def solve_instances(A_list):
#    # solve positive 1-k-SAT problem
#    num = A_list.shape[0]
#    sol_bool = np.zeros(num, dtype=int)
#    for i in range(num):
#        sol = ec.get_exact_cover(A_list[i])
#        #print('sol',sol)
#        if len(sol) > 0:
#            sol_bool[i] = 1
#    return sol_bool
#
#
def solve_instances(A):
    # solve positive 1-k-SAT problem
    sol = ec.get_exact_cover(A)
    if len(sol) > 0:
        sol_bool = 1
    else:
        sol_bool = 0
    return sol_bool
def solvekSAT(A):
    # solve k-SAT problem
    n, m = A.shape
    g = Glucose3()
    for i in range(m):
        clause = A[:,i]*np.arange(1, n+1, dtype=int)
        clause = [int(x) for x in clause if x != 0]
        g.add_clause(clause)
    return g.solve()
