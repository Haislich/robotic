# Scalars

Every number in a formula is a `Scalar`: a symbolic expression whose type the type
checker knows. sympy does the algebra underneath, but no sympy type appears in the
library's signatures, so a formula stays typed from the first symbol to the last
simplification.

## Symbols

A symbol is created from its name. Equal names give the same symbol, so two
creations of `q2` add up to `2*q2`, not to `q2 + q2` with two different variables.

```python
>>> from robotic import Scalar
>>> q1, q2 = Scalar.symbol("q1"), Scalar.symbol("q2")
>>> Scalar.symbol("q2") + q2
2*q2

```

Symbols are real. Arithmetic takes a scalar, an `int` or a `float` on either side and
always gives a scalar:

```python
>>> 2 * q1 - q2 / 3
2*q1 - q2/3
>>> 1 / q1
1/q1
>>> -q1**2
-q1**2

```

## Exact constants

Integers stay exact, and so do their quotients. Exact π, square roots and
exponentials are methods, so no exact value passes through a float:

```python
>>> Scalar.pi() / 2
pi/2
>>> Scalar(2).sqrt()
sqrt(2)
>>> Scalar(1) / 10
1/10
>>> Scalar(1).exp()
E

```

## Functions

Trigonometric functions, `atan2` (the receiver is the y coordinate), square root and
exponential are methods returning scalars:

```python
>>> q1.cos()
cos(q1)
>>> q1.sin().atan2(q1.cos())
atan2(sin(q1), cos(q1))
>>> (q1.cos() ** 2 + q1.sin() ** 2).simplify()
1
>>> q1.cos().latex()
'\\cos{\\left(q_{1} \\right)}'

```

## The float policy

A float is accepted wherever a number is, and it enters the expression as written: a
floating-point constant, never converted to a fraction. Writing `0.1` means not caring
about exactness.

```python
>>> q1 + 0.1
q1 + 0.1

```

The price is visible in symbolic results. The entry r₁₁ of Rz(q) Rz(π/2) is
cos q cos(π/2) − sin q sin(π/2). With the float `math.pi / 2`, cos(π/2) is not zero
but the rounding error `6.12323399573677e-17`, and simplification keeps it:

```python
>>> import math
>>> half_turn = Scalar(math.pi / 2)
>>> (q1.cos() * half_turn.cos() - q1.sin() * half_turn.sin()).simplify()
-1.0*sin(q1) + 6.12323399573677e-17*cos(q1)

```

With the exact constant the same entry is exactly −sin q:

```python
>>> half_turn = Scalar.pi() / 2
>>> (q1.cos() * half_turn.cos() - q1.sin() * half_turn.sin()).simplify()
-sin(q1)

```
