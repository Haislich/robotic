# Frames, positions and displacements

## Frames are types

A frame is a class you declare. How the course writes it, its notation, is a class
keyword, because the exam numbers its frames (^0, ^3') and a class cannot be called
`0`; without the keyword the class name is used.

```python
>>> from robotic import Expression, Frame, Scalar, Symbol
>>> class Base(Frame, notation="0"):
...     """Fixed frame at the first joint."""
>>> class Hand(Frame):
...     """Frame of the hand."""
>>> Base.notation, Hand.notation
('0', 'Hand')

```

A frame gives two kinds of three numbers. A **position** is where something is,
measured from the frame's origin: a point, or its position vector. A **displacement**
is how much and which way to move, read on the frame's axes, from nowhere in
particular: a free vector.

```python
>>> Hand.position(1, 2, 0)
Position[Hand](1, 2, 0)
>>> Hand.displacement(1, 2, 0)
Displacement[Hand](1, 2, 0)
>>> Hand.origin()
Position[Hand](0, 0, 0)

```

`Point` is another name for `Position`, the same class.

The frame is part of the type: `Hand.displacement(1, 0, 0)` is a
`Displacement[Scalar, Hand]`, and adding it to a displacement in `Base` is a type
error before anything runs. If the
type checker is ignored, the frames stored in the values raise at the operation:

```python
>>> Hand.displacement(1, 0, 0) + Base.displacement(1, 0, 0)
Traceback (most recent call last):
...
robotic.geometry.FrameMismatchError: cannot add Displacement[Hand] and Displacement[Base]: frame Base where Hand is expected

```

## Constant or symbolic

The first type parameter says whether the numbers are fixed. A position or
displacement is constant, `[Scalar, F]`, when every number is a constant, and symbolic,
`[Expression, F]`, as soon as one is built from a symbol; then every component is an
expression, constant ones included. `evaluate` substitutes exact values and gives the
constant one back:

```python
>>> q = Symbol("q")
>>> tip = Hand.position(1, 2, q)
>>> isinstance(tip.x, Expression)
True
>>> tip.evaluate({"q": 3})
Position[Hand](1, 2, 3)

```

## Why a position is not a displacement

Take two fingertips of a hand whose frame has the same orientation as `Base` and its
origin 5 along x. In `Hand` the fingertips are a = (1, 2, 0) and b = (3, 2, 0); in
`Base` every coordinate of a position moves by 5 along x.

```python
>>> a, b = Hand.position(1, 2, 0), Hand.position(3, 2, 0)
>>> a_base, b_base = Base.position(6, 2, 0), Base.position(8, 2, 0)

```

The displacement from a to b is the same in both frames, because the translation
cancels:

```python
>>> b - a
Displacement[Hand](2, 0, 0)
>>> b_base - a_base
Displacement[Base](2, 0, 0)

```

A sum of positions would depend on where the origin is: (1 + 3) = 4 in `Hand`, but
(6 + 8) = 14 in `Base`, not 4 + 5. No frame-independent position is "a + b", so
`a + b`, `2 * a` and `-a` are type errors. Moving a position by a displacement, or
taking the displacement between two positions, is fine:

```python
>>> a + (b - a) / 2
Position[Hand](2, 2, 0)
>>> a - Hand.origin()
Displacement[Hand](1, 2, 0)

```

The second line is how a position's coordinates become a displacement: explicitly,
as the displacement from the origin.

## The homogeneous w

The rule is the bookkeeping of the homogeneous coordinate: a position is [p; 1], a
displacement [v; 0], and a result is meaningful only when its w is 0 or 1.

| Operation | w | Result |
|---|---|---|
| position − position | 1 − 1 = 0 | displacement |
| position ± displacement | 1 ± 0 = 1 | position |
| displacement ± displacement | 0 ± 0 = 0 | displacement |
| k · displacement | k · 0 = 0 | displacement |
| position + position | 1 + 1 = 2 | type error |
| k · position | k · 1 = k | type error |

## Displacement algebra

Dot and cross products need both displacements in the same frame. The norm is a
scalar:

```python
>>> x, y = Hand.displacement(1, 0, 0), Hand.displacement(0, 1, 0)
>>> x.cross(y)
Displacement[Hand](0, 0, 1)
>>> Hand.displacement(1, 2, 3).dot(Hand.displacement(4, 5, 6))
32
>>> Hand.displacement(3, 4, 0).norm()
5

```

## Directions

A direction is a displacement of length one. It is always a normalization: written
from components, `Hand.direction(x, y, z)` divides them by their norm; computed,
`d.normalize()` does the same. The norm is not simplified; call `.simplify()` for the
compact form.

```python
>>> Hand.direction(3, 4, 0)
Direction[Hand](3/5, 4/5, 0)
>>> heading = Hand.displacement(q.cos(), q.sin(), 0).normalize()
>>> heading
Direction[Hand](cos(q)/sqrt(sin(q)**2 + cos(q)**2), sin(q)/sqrt(sin(q)**2 + cos(q)**2), 0)
>>> heading.simplify()
Direction[Hand](cos(q), sin(q), 0)

```

A constant displacement whose norm is zero, within the tolerance of `Scalar.is_close`,
has no direction, so normalizing it raises:

```python
>>> Hand.direction(0, 0, 0)
Traceback (most recent call last):
...
robotic.geometry.ZeroVectorError: cannot normalize the zero vector Displacement[Hand](0, 0, 0)

```

A symbolic displacement never raises: its norm may vanish at some configurations, or
at all of them, and the library does not decide which. There the direction is
undefined, as at any singular configuration:

```python
>>> Hand.displacement(q, 0, 0).normalize()
Direction[Hand](q/Abs(q), 0, 0)

```

## LaTeX

Positions and displacements render as columns with the frame as left superscript:

```python
>>> print(Base.displacement(q, 0, 1).latex())
{}^{0}\begin{bmatrix} q \\ 0 \\ 1 \end{bmatrix}

```
