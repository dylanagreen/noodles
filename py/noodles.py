import numpy as np

def nnls(A, y, max_iter=100, eps=1e-3):
    """
    Non-negative least squares solver in python. This is a direct implementation
    of the Lawson-Hanson algorithm. The Bro-Jong speedups are *not* implemented.
    """
    x = np.zeros(A.shape[1])
    P = np.zeros_like(x, dtype=bool)
    R = np.ones_like(x, dtype=bool)

    u = A.T @ y
    n_iter = 0
    while np.any(R) and (np.max(u[R]) > eps) and (n_iter < max_iter):
        # print(n_iter, np.max(u[R]))
        n_iter += 1
        # Add the index of maximum constraint to
        # the passive set P and remove it from the active
        # set R
        max_val = np.max(u[R])
        j = np.where(u == max_val)
        P[j] = 1
        R[j] = 0

        # Generate a prediction of the passive variables
        s = np.zeros_like(x)
        A_p = A[:, P]
        s[P] = np.linalg.lstsq(A_p, y)[0]

        while np.min(s[P]) <= 0:
            print("trig")
            # Calculate the alpha value from the passive set
            # where the variables are negative
            diffs = x / (x - s)
            alpha = np.min(diffs[P & (s <= 0)])
            x += alpha * (s - x)

            # Move any newly negative predictions from the passive set
            # to the fixed set
            x_lt_z = x <= 0
            R[x_lt_z] = 1
            P[x_lt_z] = 0

            # Generate a new prediction of the active variables
            s = np.zeros_like(x)
            A_p = A[:, P]
            s[P] = np.linalg.lstsq(A_p, y)[0]

        x = s
        u = A.T @ (y - A @ x)

    return x