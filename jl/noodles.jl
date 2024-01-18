using LinearAlgebra

function lsq(A, y)
  (transpose(A) * A) \ (transpose(A) * y)
end

function nnls(A, y, eps=1e-3)
  x = zeros(size(A)[2])
  P = zeros(Bool, size(x))
  R = ones(Bool, size(x))

  A = copy(A)
  u = transpose(A) * y
  max_iter = 100
  n_iter = 0

  while any(R) & (maximum(u[R], init=eps) > eps) & (n_iter < max_iter)
    n_iter += 1

    # Add the index of maximum constraint to
    # the active set P and remove it from the fixed
    # set R
    max_val = maximum(u[R])
    j = u .== max_val
    P[j] .= 1
    R[j] .= 0

    # Generate a prediction of the active variables
    s = zeros(size(x))
    A_p = A[:, P]
    s[P] = lsq(A_p, y)

    while minimum(s[P]) <= 0
      diffs = x ./ (x  - s)
      alpha = minimum(diffs[P & (s .<= 0)])
      x += alpha * (s - x)

      # Move any newly negative predictions from the active set
      # to the fixed set
      x_lt_z = x .<= 0
      R[x_lt_z] .= 1
      P[x_lt_z] .= 0

      # Generate a new prediction of the active variables
      s = zeros(size(x))
      A_p = A[:, P]
      s[P] = lsq(A_p, y)

    end
    x = s
    u = transpose(A) * (y - A * x)
  end
  return x
end