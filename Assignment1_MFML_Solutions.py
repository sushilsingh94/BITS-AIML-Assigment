"""
AIMLC ZC416 - Mathematical Foundations for Machine Learning
Assignment 1 Solutions
"""

import numpy as np

# No seed used — system generated random numbers as per instructions

# =============================================================================
# Q1) Finding Solutions of Linear Systems
# =============================================================================

# ---- Q1(a): REF and RREF without built-in functions ----
# We take m < n (underdetermined system - more unknowns than equations)

def row_echelon_form(augmented):
    """
    Converts an augmented matrix [A|b] to Row Echelon Form (REF).
    No built-in functions used for the row reduction.
    """
    M = augmented.astype(float).copy()
    rows, cols = M.shape
    pivot_row = 0

    for col in range(cols - 1):  # don't pivot on the last column (b)
        if pivot_row >= rows:
            break

        # Find the first non-zero entry in this column at or below pivot_row
        max_row = -1
        for r in range(pivot_row, rows):
            if abs(M[r, col]) > 1e-10:
                if max_row == -1 or abs(M[r, col]) > abs(M[max_row, col]):
                    max_row = r
                break

        if max_row == -1:
            continue  # no pivot in this column, move to next

        # Swap rows to bring pivot up
        if max_row != pivot_row:
            M[[pivot_row, max_row]] = M[[max_row, pivot_row]]

        # Eliminate entries below the pivot
        for r in range(pivot_row + 1, rows):
            if abs(M[r, col]) > 1e-10:
                factor = M[r, col] / M[pivot_row, col]
                M[r] = M[r] - factor * M[pivot_row]

        pivot_row += 1

    return M


def reduced_row_echelon_form(augmented):
    """
    Converts an augmented matrix [A|b] to Reduced Row Echelon Form (RREF).
    First gets REF, then back-substitutes to make pivots = 1 and zeros above pivots.
    """
    M = row_echelon_form(augmented)
    rows, cols = M.shape

    # Find pivot positions
    pivot_cols = []
    for r in range(rows):
        for c in range(cols - 1):
            if abs(M[r, c]) > 1e-10:
                pivot_cols.append((r, c))
                break

    # Back substitution: make each pivot 1 and eliminate above
    for (r, c) in reversed(pivot_cols):
        # Scale the pivot row so pivot becomes 1
        M[r] = M[r] / M[r, c]

        # Eliminate all entries above the pivot in this column
        for r2 in range(r):
            if abs(M[r2, c]) > 1e-10:
                factor = M[r2, c]
                M[r2] = M[r2] - factor * M[r]

    return M


# Generate a random matrix A (m x n) with m < n
m, n = 3, 5  # 3 equations, 5 unknowns (small for easy copying)
A = np.random.randint(-5, 6, size=(m, n)).astype(float)
b = np.random.randint(-5, 6, size=(m, 1)).astype(float)

print("=" * 60)
print("Q1(a): REF and RREF")
print("=" * 60)
print(f"\nMatrix A ({m}x{n}):")
print(A)
print(f"\nVector b ({m}x1):")
print(b)

# Augmented matrix [A | b]
augmented = np.hstack([A, b])
print("\nAugmented matrix [A|b]:")
print(augmented)

ref = row_echelon_form(augmented)
print("\nREF (Row Echelon Form):")
print(np.round(ref, 4))

rref = reduced_row_echelon_form(augmented)
print("\nRREF (Reduced Row Echelon Form):")
print(np.round(rref, 4))


# ---- Q1(b): Pivot columns, non-pivot columns, particular solution, Ax=0 ----

print("\n" + "=" * 60)
print("Q1(b): Pivot and Non-Pivot Columns, Particular Solution, Null Space")
print("=" * 60)

def find_pivots(rref_matrix, n_cols):
    """
    Identifies pivot and non-pivot (free) columns from the RREF.
    """
    rows = rref_matrix.shape[0]
    pivot_cols = []
    for r in range(rows):
        for c in range(n_cols):
            if abs(rref_matrix[r, c]) > 1e-10:
                pivot_cols.append(c)
                break
    free_cols = [c for c in range(n_cols) if c not in pivot_cols]
    return pivot_cols, free_cols


pivot_cols, free_cols = find_pivots(rref, n)
print(f"\nPivot columns (0-indexed): {pivot_cols}")
print(f"Non-pivot (free) columns (0-indexed): {free_cols}")
print(f"Rank of A = {len(pivot_cols)}")

# Particular solution: set all free variables to 0, read pivot values from RREF
def particular_solution(rref_matrix, pivot_cols, free_cols, n):
    """
    Find a particular solution by setting free variables to 0.
    """
    x = np.zeros(n)
    for i, pc in enumerate(pivot_cols):
        x[pc] = rref_matrix[i, -1]  # last column is b in augmented
    return x

x_particular = particular_solution(rref, pivot_cols, free_cols, n)
print(f"\nParticular solution (free variables = 0):")
print(x_particular)
print(f"Verification A @ x_p = {A @ x_particular}")
print(f"Original b = {b.flatten()}")

# Solutions to Ax = 0 (null space basis vectors)
def null_space_basis(rref_matrix, pivot_cols, free_cols, n):
    """
    For each free variable, set it to 1 (others to 0) and solve for pivot variables.
    This gives a basis for the null space of A.
    """
    basis = []
    for fc in free_cols:
        x = np.zeros(n)
        x[fc] = 1  # set this free variable to 1
        # For each pivot row, solve for the pivot variable
        for i, pc in enumerate(pivot_cols):
            x[pc] = -rref_matrix[i, fc]
        basis.append(x)
    return basis

null_basis = null_space_basis(rref, pivot_cols, free_cols, n)
print(f"\nNull space basis vectors (solutions to Ax = 0):")
for i, v in enumerate(null_basis):
    print(f"  v{i+1} = {v}")
    print(f"  Verification A @ v{i+1} = {np.round(A @ v, 10)}")


# ---- Q1(c): Random 6x9 matrix, full analysis ----

print("\n" + "=" * 60)
print("Q1(c): Complete Analysis of 6x9 System")
print("=" * 60)

m2, n2 = 6, 9
A2 = np.random.randint(-5, 6, size=(m2, n2)).astype(float)

# To ensure the system is consistent, we pick x and compute b = A2 @ x
x_true = np.random.randint(-3, 4, size=(n2, 1)).astype(float)
b2 = A2 @ x_true  # This guarantees Ax = b has at least one solution

print(f"\nRandom matrix A ({m2}x{n2}):")
print(A2)
print(f"\nVector b ({m2}x1):")
print(b2.flatten())

augmented2 = np.hstack([A2, b2])

ref2 = row_echelon_form(augmented2)
print("\nREF:")
print(np.round(ref2, 4))

rref2 = reduced_row_echelon_form(augmented2)
print("\nRREF:")
print(np.round(rref2, 4))

pivot_cols2, free_cols2 = find_pivots(rref2, n2)
print(f"\nPivot columns: {pivot_cols2}")
print(f"Non-pivot (free) columns: {free_cols2}")
print(f"Rank = {len(pivot_cols2)}")

x_part2 = particular_solution(rref2, pivot_cols2, free_cols2, n2)
print(f"\nParticular solution: {np.round(x_part2, 4)}")

null_basis2 = null_space_basis(rref2, pivot_cols2, free_cols2, n2)
print(f"\nNull space basis vectors:")
for i, v in enumerate(null_basis2):
    print(f"  v{i+1} = {np.round(v, 4)}")

# General solution: x = x_particular + c1*v1 + c2*v2 + ...
print("\nGeneral Solution:")
print("x = x_particular + c1*v1 + c2*v2 + ... + ck*vk")
print("where c1, c2, ..., ck are arbitrary real numbers.")

# Verification with random constants
c_vals = np.random.randn(len(null_basis2))
x_general = x_part2.copy()
for i, v in enumerate(null_basis2):
    x_general = x_general + c_vals[i] * v

print(f"\nVerification with random constants {np.round(c_vals, 3)}:")
print(f"  x_general = {np.round(x_general, 4)}")
print(f"  A @ x_general = {np.round(A2 @ x_general, 4)}")
print(f"  b = {b2.flatten()}")
print(f"  Difference (should be ~0): {np.round(np.linalg.norm(A2 @ x_general - b2.flatten()), 10)}")


# =============================================================================
# Q2) Matrix Decompositions
# =============================================================================

# ---- Q2(a): LU Decomposition using Elementary Matrices ----

print("\n" + "=" * 60)
print("Q2(a): LU Decomposition with Elementary Matrices")
print("=" * 60)

# Generate a symmetric positive definite matrix
def generate_spd_matrix(n):
    """
    Generates a random n x n symmetric positive definite matrix.
    Method: A = B^T B + n*I ensures positive definiteness.
    """
    B = np.random.randint(-3, 4, size=(n, n)).astype(float)
    return B.T @ B + n * np.eye(n)

n_size = 3
A_spd = generate_spd_matrix(n_size)
print(f"\nSymmetric Positive Definite matrix A ({n_size}x{n_size}):")
print(A_spd)
print(f"Symmetric check (A - A^T should be 0): {np.linalg.norm(A_spd - A_spd.T):.10f}")
print(f"Eigenvalues (all should be positive): {np.round(np.linalg.eigvals(A_spd), 4)}")

def lu_decomposition_with_elementary(A):
    """
    Performs LU decomposition by constructing elementary matrices for each
    row operation. L is found as the product of inverses of elementary matrices.
    """
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.eye(n)
    elementary_matrices = []

    for col in range(n):
        for row in range(col + 1, n):
            if abs(U[row, col]) > 1e-10:
                factor = U[row, col] / U[col, col]

                # Elementary matrix E: identity with -factor at (row, col)
                E = np.eye(n)
                E[row, col] = -factor
                elementary_matrices.append(E)

                # Apply the row operation to U
                U[row] = U[row] - factor * U[col]

                # Inverse of E: just flip the sign of the off-diagonal entry
                E_inv = np.eye(n)
                E_inv[row, col] = factor
                L = L @ E_inv

    return L, U, elementary_matrices

L, U, elem_matrices = lu_decomposition_with_elementary(A_spd)

print(f"\nElementary matrices used:")
for i, E in enumerate(elem_matrices):
    print(f"  E{i+1}:")
    print(f"  {E}")

print(f"\nLower triangular matrix L:")
print(np.round(L, 4))

print(f"\nUpper triangular matrix U:")
print(np.round(U, 4))

print(f"\nVerification: L @ U =")
print(np.round(L @ U, 4))

print(f"\nOriginal A =")
print(np.round(A_spd, 4))

print(f"\nDifference ||A - LU|| = {np.linalg.norm(A_spd - L @ U):.10f}")


# ---- Q2(b): Cholesky Decomposition ----

print("\n" + "=" * 60)
print("Q2(b): Cholesky Decomposition")
print("=" * 60)

def cholesky_decomposition(A):
    """
    Computes Cholesky decomposition A = L @ L^T for a symmetric positive definite matrix.
    L is lower triangular.
    """
    n = A.shape[0]
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                # Diagonal elements
                sum_sq = sum(L[i, k] ** 2 for k in range(j))
                L[i, j] = np.sqrt(A[i, i] - sum_sq)
            else:
                # Off-diagonal elements
                sum_prod = sum(L[i, k] * L[j, k] for k in range(j))
                L[i, j] = (A[i, j] - sum_prod) / L[j, j]

    return L

L_chol = cholesky_decomposition(A_spd)

print(f"\nUsing the same SPD matrix A from Q2(a):")
print(A_spd)

print(f"\nCholesky factor L (lower triangular):")
print(np.round(L_chol, 4))

print(f"\nL^T:")
print(np.round(L_chol.T, 4))

print(f"\nVerification: L @ L^T =")
print(np.round(L_chol @ L_chol.T, 4))

print(f"\nDifference ||A - L @ L^T|| = {np.linalg.norm(A_spd - L_chol @ L_chol.T):.10f}")


# ---- Q2(c): QR Decomposition ----

print("\n" + "=" * 60)
print("Q2(c): QR Decomposition (m > n, linearly independent columns)")
print("=" * 60)

# m > n: more rows than columns, columns are linearly independent
m_qr, n_qr = 5, 3
# Generate a matrix with linearly independent columns
A_qr = np.random.randint(-5, 6, size=(m_qr, n_qr)).astype(float)
# Check rank
while np.linalg.matrix_rank(A_qr) < n_qr:
    A_qr = np.random.randint(-5, 6, size=(m_qr, n_qr)).astype(float)

print(f"\nMatrix A ({m_qr}x{n_qr}) with {n_qr} linearly independent columns:")
print(A_qr)
print(f"Rank of A: {np.linalg.matrix_rank(A_qr)}")

def qr_decomposition(A):
    """
    QR decomposition using Gram-Schmidt orthogonalization.
    A = Q @ R where Q has orthonormal columns and R is upper triangular.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        # Subtract projections onto previous q vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

Q, R = qr_decomposition(A_qr)

print(f"\nOrthogonal matrix Q ({m_qr}x{n_qr}):")
print(np.round(Q, 4))

print(f"\nUpper triangular matrix R ({n_qr}x{n_qr}):")
print(np.round(R, 4))

print(f"\nVerification: Q @ R =")
print(np.round(Q @ R, 4))

print(f"\nDifference ||A - QR|| = {np.linalg.norm(A_qr - Q @ R):.10f}")

print(f"\nOrthonormality check Q^T @ Q (should be identity):")
print(np.round(Q.T @ Q, 4))


# ---- Q2(d): 7x5 Random Matrix QR, Diagonal of R ----

print("\n" + "=" * 60)
print("Q2(d): 7x5 Matrix QR Decomposition - Diagonal Elements of R")
print("=" * 60)

m_d, n_d = 7, 5
A_d = np.random.randint(-5, 6, size=(m_d, n_d)).astype(float)
# Make sure columns are linearly independent
while np.linalg.matrix_rank(A_d) < n_d:
    A_d = np.random.randint(-5, 6, size=(m_d, n_d)).astype(float)

print(f"\nRandom matrix A ({m_d}x{n_d}) with linearly independent columns:")
print(A_d)

Q_d, R_d = qr_decomposition(A_d)

print(f"\nQ matrix ({m_d}x{n_d}):")
print(np.round(Q_d, 4))

print(f"\nR matrix ({n_d}x{n_d}):")
print(np.round(R_d, 4))

print(f"\nDiagonal elements of R: {np.round(np.diag(R_d), 4)}")

print(f"\nVerification ||A - QR|| = {np.linalg.norm(A_d - Q_d @ R_d):.10f}")

print("""
Observation on diagonal elements of R:
---------------------------------------
All diagonal elements of R are strictly positive (non-zero). This is because
the columns of A are linearly independent. In the Gram-Schmidt process,
R[j,j] = ||v_j|| where v_j is the component of the j-th column of A that is
orthogonal to the span of previous columns. Since the columns are linearly
independent, no column lies in the span of the previous ones, so this orthogonal
component is never zero. Hence all diagonal entries of R are positive.

If any diagonal element were zero, it would mean that column was a linear
combination of the previous columns, contradicting linear independence.
""")
