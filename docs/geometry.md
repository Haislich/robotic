# Frames, points and vectors

## Frames are types

A frame is a class you declare. How the course writes it, its notation, is a class
keyword, because the exam numbers its frames (^0, ^3') and a class cannot be called
`0`; without the keyword the class name is used.

```python
>>> from robotic import Frame, Scalar
>>> class Base(Frame, notation="0"):
...     """Fixed frame at the first joint."""
>>> class Hand(Frame):
...     """Frame of the hand."""
>>> Base.notation, Hand.notation
('0', 'Hand')

```

A frame gives a point its origin and axes, and a vector its axes only:

```python
>>> Hand.point(1, 2, 0)
Point[Hand](1, 2, 0)
>>> Hand.vector(1, 2, 0)
Vector[Hand](1, 2, 0)
>>> Hand.origin()
Point[Hand](0, 0, 0)

```

The frame is part of the type: `Hand.vector(1, 0, 0)` is a `Vector[Hand]`, and adding
it to a `Vector[Base]` is a type error before anything runs.

## Why a point is not a vector

Take two fingertips of a hand whose frame has the same orientation as `Base` and its
origin 5 along x. In `Hand` the fingertips are a = (1, 2, 0) and b = (3, 2, 0); in
`Base` every coordinate of a point moves by 5 along x.

```python
>>> a, b = Hand.point(1, 2, 0), Hand.point(3, 2, 0)
>>> a_base, b_base = Base.point(6, 2, 0), Base.point(8, 2, 0)

```

The vector from a to b is the same arrow in both frames, because the translation
cancels:

```python
>>> b - a
Vector[Hand](2, 0, 0)
>>> b_base - a_base
Vector[Base](2, 0, 0)

```

A sum of points would depend on where the origin is: (1 + 3) = 4 in `Hand`, but
(6 + 8) = 14 in `Base`, not 4 + 5. No frame-independent point is "a + b", so `a + b`,
`2 * a` and `-a` are type errors. Moving a point by a vector, or taking the vector
between two points, is fine:

```python
>>> a + (b - a) / 2
Point[Hand](2, 2, 0)
>>> a - Hand.origin()
Vector[Hand](1, 2, 0)

```

The second line is how coordinates become a vector: explicitly, as the grounded vector
from the origin.

## The homogeneous w

The rule is the bookkeeping of the homogeneous coordinate: a point is [p; 1], a
vector [v; 0], and a result is meaningful only when its w is 0 or 1.

| Operation | w | Result |
|---|---|---|
| point − point | 1 − 1 = 0 | vector |
| point ± vector | 1 ± 0 = 1 | point |
| vector ± vector | 0 ± 0 = 0 | vector |
| k · vector | k · 0 = 0 | vector |
| point + point | 1 + 1 = 2 | type error |
| k · point | k · 1 = k | type error |

## Vector algebra

Dot and cross products need both vectors in the same frame. The norm is a scalar:

```python
>>> x, y = Hand.vector(1, 0, 0), Hand.vector(0, 1, 0)
>>> x.cross(y)
Vector[Hand](0, 0, 1)
>>> Hand.vector(1, 2, 3).dot(Hand.vector(4, 5, 6))
32
>>> Hand.vector(3, 4, 0).norm()
5

```

A unit vector is only ever the normalization of a vector: building one divides the
components by their simplified norm, so no square root is left where none is needed.

```python
>>> Hand.vector(3, 4, 0).normalize()
UnitVector[Hand](3/5, 4/5, 0)
>>> q = Scalar.symbol("q")
>>> Hand.vector(q.cos(), q.sin(), 0).normalize()
UnitVector[Hand](cos(q), sin(q), 0)

```

The zero vector has no direction, so normalizing it raises:

```python
>>> Hand.vector(0, 0, 0).normalize()
Traceback (most recent call last):
...
robotic.geometry.ZeroVectorError: cannot normalize the zero vector Vector[Hand](0, 0, 0)

```

## LaTeX

Points and vectors render as columns with the frame as left superscript:

```python
>>> print(Base.vector(q, 0, 1).latex())
{}^{0}\begin{bmatrix} q \\ 0 \\ 1 \end{bmatrix}

```
