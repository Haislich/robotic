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

Symbols are real. Arithmetic takes a scalar or an `int` on either side and always
gives a scalar:

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

## Exact numbers only

A symbolic value takes exact numbers only: integers, exact constants and symbols. A
float is a number near the value meant, not the value: `math.pi` is not π. It is a type
error, and it raises when executed:

```python
>>> import math
>>> Scalar(math.pi)
Traceback (most recent call last):
...
TypeError: a symbolic scalar takes an exact number, an int or a Scalar, not float 3.141592653589793; write Scalar(3) / 10 for 0.3 and Scalar.pi() for pi

```

So results print as the mathematics is written. The entry r₁₁ of Rz(q) Rz(π/2) is
cos q cos(π/2) − sin q sin(π/2), exactly −sin q:

```python
>>> half_turn = Scalar.pi() / 2
>>> (q1.cos() * half_turn.cos() - q1.sin() * half_turn.sin()).simplify()
-sin(q1)

```

Floats belong to the numeric side: the values a compiled function takes, and the
numbers it returns (see `compilation.md`).

## No equality

Whether two symbolic expressions are equal on the whole domain cannot be known in
general, so a scalar has no `==`, no hash and no truth value: each raises instead of
answering by accident.

```python
>>> q1 == q1
Traceback (most recent call last):
...
TypeError: symbolic values have no equality: whether two of them are equal on the whole domain cannot be known; compile them and compare the numbers at a configuration

```

To compare, simplify and read the result, or compile both and compare the numbers at
a configuration.
