# noodles
Straightforward NNLS solvers in Python &amp; Julia

It's called noodles because when I look at "NNLS" (Non-negative least squares)
I think of the word noodles.

## What is it?
This repo implements the Lawson-Hanson algorithm for solving NNLS problems
in both python and julia. The Python version requires numpy but
is otherwise pure python, the Julia version
requires no external packages.

These implementations are very rudimentary and straightforward. There are some
optimizations to be made but not yet implemented (TBI, to be implemented).

The Bro-Jong (1999) speedups are not implemented here, as they only
become useful when the number of iterations gets high (which is vaguely
correlated with how many elements are in your x matrix, or correspondingly
the number of templates to fit). They can also be useful when solving
for many fits at once (TBI, maybe), so its not out of the question to include
them here.

## How fast is it?
Brief timing tests are in the `/nb` folder.



