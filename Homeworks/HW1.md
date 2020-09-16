## Intro to ML | Homework 1

# I. List 3 reasons why you, personally, want to learn linear algebra

- Basic knowledge never hurts.
- It's indispensable for ML
- Neccessary for any Data science field

# II. Find five quotes from research papers, blogs, or books that define the field of linear algebra.

- [Linear algebra is an area of study in mathematics that concerns itself primarily with the study of vector spaces and the linear transformations between them.](https://brilliant.org/wiki/linear-algebra/)
  &nbsp;

- [Linear algebra is a field of mathematics that is universally agreed to be a prerequisite to a deeper understanding of machine learning.](https://machinelearningmastery.com/gentle-introduction-linear-algebra/)
  &nbsp;

- [The branch of mathematics that deals with the theory of systems of linear equations, matrices, vector spaces, determinants, and linear transformations](https://www.yourdictionary.com/linear-algebra)
  &nbsp;

- [Linear Algebra is a continuous form of mathematics and is applied throughout science and engineering because it allows you to model natural phenomena and to compute them efficiently.](https://towardsdatascience.com/linear-algebra-for-deep-learning-f21d7e7d7f23)
  &nbsp;

- [Linear algebra is the study of vectors and linear functions.](https://www.math.ucdavis.edu/~linear/linear-guest.pdf)

# III. Implement other vector arithmetic operations

#### such as addition, division, subtraction, and the vector dot product

&nbsp;
Vector addition:

```python
a + b = (a1 + b1, a2 + b2, ..., an + bn)
```

&nbsp;
Vector substraction:

```python
a - b = (a1 - b1, a2 - b2, ..., an - bn)
```

&nbsp;
Vector dot product:

```python
a • b = (|a| * |b|) / cos𝞱
```

where `cos𝞱` is the angle between `a` and `b` vectors

&nbsp;
Vector division:
_Division of vectors is not defined._

# IV. Implement more matrix arithmetic operations

####such as subtraction, division, the Hadamard product, and vector-matrix multiplication
&nbsp;
Matrix substraction:

```python
A - B = [[a(1, 1) - B(1, 1), ..., A(m, 1) - B(m, 1)],
         ...
         [a(1, n) - B(1, n), ..., A(m, n) - B(m, n)]]
```

&nbsp;
Hadamard product (elementwise multiplication)

```python
A ∘ B = [[a(1, 1) * B(1, 1), ..., A(m, 1) * B(m, 1)],
         ...
         [a(1, n) * B(1, n), ..., A(m, n) * B(m, n)]]
```

&nbsp;
Matrix division

```python
A / B = A * B⁻¹
```

# V. Develop examples for other matrix operations

#### such as the determinant, trace, and rank

&nbsp;
Determinant of a matrix

```python
         n
det(A) = ∑ (-1)¹⁺ʲ * a(1,j) * det(A(1,j))
        j=1
```

where `n` is the number of columns and rows in the matrix

_example (2 by 2 matrix)_

```python
A = [[8, 15],
     [7, -3]]

det(A) = (8 * (-3)) - (7 * 15) = -129
```

_example (3 by 3 matrix)_

```python
A = [[2, -3,  1],
     [2,  0, -1],
     [1,  4,  5]]

det(A) = 2 * det([[0, -1], [4, 5]]) - (-3) * det([[2, -1], [1, 5]]) + 1 * det([[2, 0], [1, 4]]) = 49
```

&nbsp;
Trace of a matrix

```python
           n
trace(A) = ∑ A(j,j)
          j=1
```

where `n` is the number of columns and rows in the matrix

```python
A = [[2,  6, 7],
     [6, -6, 8],
     [0,  2, 4]]

trace(A) = 2 + (-6) + 4 = 0
```

&nbsp;
Rank of a matrix

[The rank of a matrix is the dimension of the vector space generated (or spanned) by its columns.](<https://en.wikipedia.org/wiki/Rank_(linear_algebra)>)

_example_

```python
A = [[1, 2, 4],
     [2, 4, 8]]
```

Submatrices:

```python
A1 = [[1, 2], [2, 4]]
A2 = [[1, 4], [2, 8]]
A3 = [[2, 4], [4, 8]]

det(A1) = 0 & det(A2) = 0 & det(A3) = 0 ⇒ rank(A) = 1
```

# VI. Implement small examples of other simple methods for matrix factorization

#### such as the QR decomposition, the Cholesky decomposition, and the eigendecomposition

&nbsp;
QR decomposition

`A = QR`
where `Q` is an `m×n` matrix with orthonormal columns (if `m ≠ n`)
and `R` is an `n×n`, upper triangular matrix.

```python
A = [[-1, -1, 1],
     [ 1,  3, 3],
     [-1, -1, 5],
     [ 1,  3, 7]]
```

thus

```python
Q = [[-1/2, 1/2, -1/2],
     [ 1/2, 1/2, -1/2],
     [-1/2, 1/2,  1/2],
     [ 1/2, 1/2,  1/2]]

R = [[2, 4, 2],
     [0, 2, 8],
     [0, 0, 4]]
```

&nbsp;
Cholesky decomposition

`A = LL*`
where `L` is an lower triangular matrix
and `L*` is the conjugate transpose of `L`

```python
A = [[  4,  12, -16],
     [ 12,  37, -43],
     [-16, -43,  98]]
```

thus

```python
L = [[ 2, 0, 0],
     [ 6, 1, 0],
     [-8, 5, 3]]

L* = [[ 2, 6, -8],
      [ 0, 1,  5],
      [ 0, 0,  3]]
```

&nbsp;
Eigendecomposition

`Av = λv`

```python
A = [[2,  2]
     [5, -1]]
```

The eigenvalues are those `λ` for which `det(A − λI) = 0`

```python
det(A − λI) = det([[2, 2], [5, -1]] - [[λ, 0], [0, λ]])
            = [[2 - λ, 2], [5, -1-λ]]
            = (2 - λ)(-1 - λ) - 10
            = λ² - λ - 12
```

Thus, the eigenvalues are the solutions for the quadtratic equation `λ² - λ - 12`, which are `λ₁ = −3` and `λ₂ = 4`

Using this solutions we can get the eigenvectors of `A` by solving `Ax = λx` for each lamdas.

First, `λ = -3`:

```python
[[2x₁ + 2x₂], [5x₁ - x₂]] = [[-3x₁], [-3x₂]]

=> 2x₁ + 2x₂ = -3x₁
   5x₁ - x₂ = -3x₂

=> 5x₁ = -2x₂
=> x₁ = -(2/5)x₂
```

so the first eigenvector can be written as
```python
u₁ = [ 2]
     [-5]
```

`λ = 4`:

```python
[[2x₁ + 2x₂], [5x₁ - x₂]] = [[4x₁], [4x₂]]

=> 2x₁ + 2x₂ = 4x₁
   5x₁ - x₂ = 4x₂

=> x₁ = x₂
```

Thus, the eigenvectors with `λ = 4` are spanned by

```python
u₁ = [1]
     [1]
```


# VII. List 5 applications of Single Value Decomposition

- Image compression
- EOF analysis
- Principal Components Analysis
- Spectral clustering
- Background removal from videos **(🤯)**
