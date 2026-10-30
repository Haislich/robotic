# Scalars

Every number in a formula is either a constant, a `Scalar`, or an `Expression`, a value
built from at least one symbol. sympy does the algebra underneath, but no sympy type
appears in the library's signatures, so a formula stays typed from the first symbol to
the last simplification.

The two kinds answer different questions. A constant has a fixed value, so asking
whether it is zero, or equal to another constant, has an answer. An expression has no
value until its symbols get one: whether it is zero is not a question the library
answers, because no method decides it in every case.

## Symbols

A symbol is a variable, created from its name. Equal names give the same symbol, so
two creations of `q2` add up to `2*q2`, not to `q2 + q2` with two different variables.

```python
>>> from robotic import Expression, Scalar, Symbol
>>> q1, q2 = Symbol("q1"), Symbol("q2")
>>> Symbol("q2") + q2
2*q2

```

Symbols are real. A `Symbol` is an `Expression`; every operation on it gives an
expression:

```python
>>> 2 * q1 - q2 / 3
2*q1 - q2/3
>>> 1 / q1
1/q1
>>> isinstance(-q1**2, Expression)
True

```

## Constant or expression

An operation gives a constant only when every operand is a constant or an `int`. As
soon as a symbol enters, the result is an expression, whatever it simplifies to:

```python
>>> isinstance(Scalar(2) + 3, Scalar)
True
>>> isinstance(Scalar(2) * q1, Expression)
True
>>> q1 - q1
0
>>> isinstance(q1 - q1, Expression)
True

```

The same rule holds for positions, displacements, rotations and transforms: one
symbolic component makes the whole quantity symbolic (see `geometry.md`).

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
exponential are methods, giving a constant on a constant and an expression on an
expression:

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

Simplification runs only when you call it. It is costly and changes only the form, so
no operation, normalization, axis-angle extraction, compilation or LaTeX rendering
does it for you: what you see is what was computed.

## Exact numbers only

A symbolic value takes exact numbers only: integers, exact constants and symbols. A
float is a number near the value meant, not the value: `math.pi` is not π. It is a type
error, and it raises when executed:

```python
>>> import math
>>> Scalar(math.pi)
Traceback (most recent call last):
...
TypeError: a symbolic scalar takes an exact number, an int, a Scalar or an Expression, not float 3.141592653589793; write Scalar(3) / 10 for 0.3 and Scalar.pi() for pi

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

## Comparing constants

Two constants compare with a tolerance: `is_close` evaluates their difference to 30
significant digits and accepts it within `1e-9`, or the tolerance passed. No
simplification is needed, which matters because simplification can miss a zero even
on constants: cos(π/7) + cos(3π/7) + cos(5π/7) is exactly 1/2.

```python
>>> seventh = Scalar.pi() / 7
>>> total = seventh.cos() + (3 * seventh).cos() + (5 * seventh).cos()
>>> total.is_close(Scalar(1) / 2)
True
>>> Scalar(1).is_close(1 + Scalar(1) / 1000, tolerance=1e-2)
True

```

An expression has no `is_close`: its value is not fixed. Evaluate it first.

## Evaluating expressions

`evaluate` substitutes exact values for the symbols, by name, and gives the constant.
Values for symbols that do not occur are allowed, so one dictionary of joint values
serves every quantity; a symbol without a value raises.

```python
>>> (2 * q1 + q2).evaluate({"q1": 1, "q2": Scalar(1) / 2, "q3": 7})
5/2
>>> (q1 - q1).evaluate()
0
>>> (2 * q1).evaluate()
Traceback (most recent call last):
...
robotic.symbolic.MissingParameterError: no value given for q1

```

A constant is always a real number. An operation or an evaluation whose result would
be undefined, infinite or complex raises `UndefinedValueError` at once, instead of
handing back `nan` (`zoo` is sympy's complex infinity):

```python
>>> Scalar(1) / 0
Traceback (most recent call last):
...
robotic.symbolic.UndefinedValueError: zoo is not a real number
>>> (q1.sin() / q1).evaluate({"q1": 0})
Traceback (most recent call last):
...
robotic.symbolic.UndefinedValueError: nan is not a real number at q1 = 0

```

## No equality

No symbolic value has `==`, a hash or a truth value: each raises instead of answering
by accident. Constants compare with `is_close`; expressions are evaluated, or compiled
and compared at a configuration.

```python
>>> q1 == q1
Traceback (most recent call last):
...
TypeError: symbolic values have no equality: compare constants with is_close; evaluate expressions, or compile them and compare the numbers at a configuration

```
