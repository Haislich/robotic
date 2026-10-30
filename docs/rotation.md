# Rotations

## ${}^A R_B$

${}^A R_B$ maps a displacement expressed in the source B to the same vector expressed in the
target A: ^A v = ^A R_B ^B v. Its columns are the axes of B resolved in A. The code
names the frames that way: `target=A`, `source=B`, `Rotation[K, A, B]`, so a chain
reads like the composition ^A R_B ^B R_C = ^A R_C. The first parameter, `K`, says
whether the rotation is constant (`Scalar`) or built from a symbol (`Expression`).

```python
>>> from robotic import Frame, Rotation, Scalar, Symbol
>>> class Base(Frame, notation="0"):
...     """Fixed frame."""
>>> class Link1(Frame, notation="1"):
...     """Frame of the first link."""
>>> class Tool(Frame):
...     """Frame of the tool."""
>>> q1, q2 = Symbol("q1"), Symbol("q2")
>>> base_link1 = Rotation.about_z(q1, target=Base, source=Link1)
>>> base_link1
Rotation[Base, Link1]([cos(q1), -sin(q1), 0], [sin(q1), cos(q1), 0], [0, 0, 1])
>>> print(base_link1.latex())
{}^{0}R_{1} = \begin{bmatrix} \cos{\left(q_{1} \right)} & - \sin{\left(q_{1} \right)} & 0 \\ \sin{\left(q_{1} \right)} & \cos{\left(q_{1} \right)} & 0 \\ 0 & 0 & 1 \end{bmatrix}

```

Entries are counted from 1, as the course writes r₁₃; an index outside 1 to 3 is a
type error, and raises when executed.

```python
>>> base_link1.entry(1, 2)
-sin(q1)

```

`about_x`, `about_y` and `about_z` build the elementary rotations, `identity` the
identity, and `about_axis` the rotation about a direction, whose frame is the
reference frame:

```python
>>> axis = Base.direction(0, 0, 1)
>>> Rotation.about_axis(axis, q1, source=Link1)
Rotation[Base, Link1]([cos(q1), -sin(q1), 0], [sin(q1), cos(q1), 0], [0, 0, 1])

```

## Composition, inverse and displacements

`@` chains inner frames, ^A R_B ^B R_C = ^A R_C, and the inverse swaps them. A chain
whose inner frames differ is a type error.

```python
>>> link1_tool = Rotation.about_x(q2, target=Link1, source=Tool)
>>> base_tool = base_link1 @ link1_tool
>>> (base_tool.inverse() @ base_tool).simplify()
Rotation[Tool, Tool]([1, 0, 0], [0, 1, 0], [0, 0, 1])

```

On a displacement of B, `R @ v` gives the same displacement resolved in A:

```python
>>> quarter_turn = Rotation.about_z(Scalar.pi() / 2, target=Base, source=Tool)
>>> quarter_turn @ Tool.displacement(1, 0, 0)
Displacement[Base](0, 1, 0)

```

A rotation does not act on positions: it relates axes and says nothing about origins.
Either write the coincident-origin assumption down, `R.homogeneous() @ p`, or resolve
the displacement from the origin, `R @ (p - Tool.origin())`.

A product of elementary rotations passes through frames the course writes RF', RF'';
declare them like any frame:

```python
>>> class Prime(Frame, notation="'"):
...     """Frame after the first elementary rotation."""
>>> wrist = Rotation.about_z(q1, target=Base, source=Prime) @ Rotation.about_x(
...     q2, target=Prime, source=Tool
... )
>>> wrist.entry(3, 3)
cos(q2)

```

## Axis-angle

`to_axis_angle` extracts both solutions symbolically, without evaluating any symbol,
following the course convention with the angle in (−π, π]:

| Case | First | Second |
|---|---|---|
| regular | (r, θ), θ in (0, π) | (−r, −θ) |
| half turn, θ = π | (r, π) | (−r, π) |
| identity, θ = 0 | (a, 0), a free | (−a, 0) |

The axis satisfies R r = r, so it has the same components in A and in B.

```python
>>> pi = Scalar.pi()
>>> solutions = Rotation.about_z(pi / 3, target=Base, source=Tool).to_axis_angle()
>>> solutions.first
AxisAngle(axis=Direction[Base](0, 0, 1), angle=pi/3)
>>> solutions.second
AxisAngle(axis=Direction[Base](0, 0, -1), angle=-pi/3)

```

Which case applies depends on the kind of the rotation. A constant rotation is
resolved on its values, at the tolerance of `Scalar.is_close`. A symbolic rotation gets
the regular formulas, whatever values the symbols will take: the symbolic side does
not have the values, so it does not predict where the formulas break. Where they do,
the axis is undefined, as at any singular configuration.

```python
>>> solutions = base_link1.to_axis_angle()
>>> solutions.first
AxisAngle(axis=Direction[Base](0, 0, sin(q1)/Abs(sin(q1))), angle=atan2(Abs(sin(q1)), cos(q1)))

```

A product that equals a constant is still symbolic: ^0 R_1 ^1 R_0 is the identity for
every q1, but it is built from q1, so it gets the regular formulas, whose skew part
vanishes everywhere. Evaluate it first to reach the exact case:

```python
>>> product = base_link1 @ base_link1.inverse()
>>> product.to_axis_angle().first
AxisAngle(axis=Direction[Base](nan, nan, nan), angle=atan2(0, sin(q1)**2 + cos(q1)**2))
>>> product.evaluate({"q1": 1}).to_axis_angle().first
AxisAngle(axis=FreeAxis[Base](0, 0, 1), angle=0)

```

Evaluating the regular formulas where they break raises, rather than giving a `nan`
axis: at q1 = 0 the skew part of ^0 R_1 vanishes.

```python
>>> solutions.evaluate({"q1": 0})
Traceback (most recent call last):
...
robotic.symbolic.UndefinedValueError: the value is undefined (nan) at q1 = 0

```

A constant half turn and the identity are recognized exactly. For the identity any axis is
valid; it is a `FreeAxis`, z unless you choose another:

```python
>>> Rotation.about_x(pi, target=Base, source=Tool).to_axis_angle().first
AxisAngle(axis=Direction[Base](1, 0, 0), angle=pi)
>>> identity = Rotation.identity(target=Base, source=Tool)
>>> identity.to_axis_angle().first
AxisAngle(axis=FreeAxis[Base](0, 0, 1), angle=0)
>>> identity.to_axis_angle(free_axis=Base.direction(1, 0, 0)).first
AxisAngle(axis=FreeAxis[Base](1, 0, 0), angle=0)

```

Axis-angle solutions compile like any quantity; see [compilation](compilation.md).
