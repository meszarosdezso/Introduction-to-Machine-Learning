## Intro to ML | Homework 2

# I. A committee of four is chosen at random from 5 married couples.

### What is the probability that the committee will not include a husband and wife?

There are `10 * 9 * 8 * 7 (5040)` ways to pick 4 people randomly out of 10, that's the total number of cases.
To select the 4 people, so that none of them has their husband/wife in that group, after selecting the first we can choose the next one from 8 more people, because the 9th is the partner of the first one. Then, we already have 2 people in the committee, meaning that they both have their partner in the 8 remaining people, so we can only choose from 6 people. And it's the same for the last one, 3 of the remaining 7 people have their partners in the committee already, so that leaves 4 people to choose from.

That means,

```python
P = (10 * 8 * 6 * 4) / (10 * 9 * 8 * 7) = 38.09%
```

# II. Compute the probability that a produced unit has ...

```
P(A) = 0.1
P(B) = 0.2

P(A⋂B) = 0.05
```

a) at least on of the defects (`A` and `B` are NOT mutually exclusive) (`P(A⋃B)`)

`P(A) + P(B) - P(A⋂B) = 0.1 + 0.2 - 0.05 = 0.25`

b) defect `A`, but not defect `B`

```python
P(A⋂B*) = P(A) - P(A⋂B) = 0.1 - 0.05 = 0.05
```

c) none of the defects

```python
P(A*⋂B*) = 1 - P(A⋃B) = 0.75
```

d) precisely one of the defects

```python
P(A⋂B*) + P(B⋂A*) = (P(A) - P(A⋂B)) + P(B) - P(A⋂B) = 0.05 + 0.15 = 0.2
```

# III. The two events `A` and `B` have both positive possibilites

a) If they are disjoint, can they be independent? `False`

b) If they are independent, can they be disjoint? `False`

# IV. Two defective units have accidentally been mixed in among three flawless units.

### In order to find the defective units, the units are tested one at a time, in order, until either the two defective units have been found, or the three flawless units have been found.

a) Determine the distribution for `X`, the number of tested units

```py
X = [2, 3, 4] # the possible cases

P(2) = 0.4 * 0.25 = 0.1
# There's a 40% chance that we choose a defective unit
# at first, and there's a 25% chance that we choose the
# other defective next
# Cases: DD

P(3) = 3/5 * 2/4 * 1/3 + 2/5 * 3/4 * 1/3 + 3/5 * 2/4 * 1/3 = 0.3
# Cases: FFF FDD DFD (each having a 10% chance)

P(4) = 6 * 0.1 =  0.6
# Cases: FDFF FDFD FFDD FFDF DFFF DFFD (each having a 10% chance)
```

b) Compute the expected value `E(X)` and the standard deviation
`SD(X)`.

```py
E(X) = Σ(Xᵢ * P(Xᵢ)) = 4 * 0.6 + 3 * 0.3 + 2 * 0.1 = 3.5

μ = (1 * 2 + 3 * 3 + 6 * 4) / 10 = 3.5 = E(X)
SD(X) = √( (Σ(Xᵢ - μ)²) / 10 ) # where Xᵢ is the current member of the population
      = √(0.45)
```

# V. Two random variables `X` and `Y`

### have expected values `E(X) = –1` and `E(Y) = 1`, and variances `V(X) = 2` and `V(Y) = 4`. The covariance is `Cov(X, Y) = –1`.

Compute the standard deviation `SD(X – 2Y + 1)`.

```py
Var(X) = Σ(x²p − μ²)
       = Σ(x²p − 1)
       = 2

Var(Y) = Σ(y²p − μ²)
       = Σ(y²p − 1)
       = 4

Cov(X, Y) = E(X - E(X) * Y - E(Y))
          = E(X + 1 * Y - 1)
          = -1

SD(X - 2Y + 1) = # I wasn't able to solve this one.
```
