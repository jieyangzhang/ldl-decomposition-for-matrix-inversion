import numpy as np
import time
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=8,suppress=True)

M = 4

def LDLH(A):
    U = np.zeros_like(A)
    L = np.zeros_like(A)
    D = np.zeros_like(A)

    for i in range(M):
        L[i, i] = 1

    for j in range(M):
        tmp_sum = 0
        for m in range(j):
            tmp_sum += U[j, m] * L[j, m].conj()
        D[j, j] = A[j, j] - tmp_sum
        tmp_sum = 0
        for i in range(j+1, M):
            tmp_sum = 0
            for m in range(j):
                tmp_sum += U[i, m] * L[j, m].conj()
            U[i, j] = A[i, j] - tmp_sum
            L[i, j] = U[i, j] / D[j, j]

    return L, D

def LDL_solve(L, D):
    M = L.shape[0]
    B = np.zeros_like(L)
    V = np.zeros_like(L)
    for i in range(M):
        B[i, i] = 1
    for j in range(M-1):
        B[j+1, j] = -L[j+1, j]
        for i in range(j+2, M):
            tmp_sum = 0
            for m in range(j+1, i):
                tmp_sum += B[m, j] * L[i, m]
            B[i, j] = -L[i, j] - tmp_sum

    for j in range(M):
        V[M-1, j] = B[M-1, j] / D[M-1, M-1]
        V[j, M-1] = V[M-1, j].conj()
        for i in range(M-2, j-1, -1):
            tmp_sum = 0
            for m in range(i+1, M):
                tmp_sum += L[m, i].conj() * V[m, j]
            V[i, j] = B[i, j] / D[i, i] - tmp_sum
            V[j, i] = V[i, j].conj()

    return V

A = np.array([[3,      2+4j,   4+2.5j,   2+7.2j], 
              [2-4j,   4   ,   7+0.4j,   22+4j], 
              [4-2.5j, 7-0.4j, 5,        42+0.22j],
              [2-7.2j, 22-4j,  42-0.22j, 6]])


V_np = np.linalg.inv(A)

L, D = LDLH(A)
V = LDL_solve(L, D)


print('v_np - v:')
print(V_np - V)

print('AV')
print(np.dot(A, V))
