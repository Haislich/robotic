# Transforms

^A T_B maps points expressed in the source B to the target A. It is the rotation
^A R_B and the translation ^A p, the origin of B expressed in A: the homogeneous matrix
[R | t; 0 0 0 1]. On a point it is R p + t; on a vector it is R v, because a
translation does not move an arrow.

## A planar 2R arm, end to end

Two revolute joints about z, links of length l1 and l2. Frame `Link1` turns by q1 at
the base; frame `Tool` turns by q2 at the end of the first link.

```python
>>> from robotic import Frame, Rotation, Scalar, Transform
>>> class Base(Frame, notation="0"):
...     """Fixed frame at the first joint."""
>>> class Link1(Frame, notation="1"):
...     """Frame moving with the first link."""
>>> class Tool(Frame, notation="2"):
...     """Frame moving with the second link."""
>>> q1, q2 = Scalar.symbol("q1"), Scalar.symbol("q2")
>>> l1, l2 = Scalar.symbol("l1"), Scalar.symbol("l2")

```

A transform is built from its two frames, which gives the identity between them.
Setting the rotation or the translation returns a new transform with the same frames,
like filling in the exam's [R | t]; a rotation between other frames, or a translation
in another frame, is a type error:

```python
>>> base_link1 = Transform(target=Base, source=Link1).with_rotation(
...     Rotation.about_z(q1, target=Base, source=Link1)
... )
>>> link1_tool = (
...     Transform(target=Link1, source=Tool)
...     .with_rotation(Rotation.about_z(q2, target=Link1, source=Tool))
...     .with_translation(Link1.vector(l1, 0, 0))
... )
>>> link1_tool.translation
Vector[Link1](l1, 0, 0)

```

Composition chains the inner frames, ^0 T_1 ^1 T_2 = ^0 T_2:

```python
>>> base_tool = base_link1 @ link1_tool
>>> base_tool.entry(1, 4)
l1*cos(q1)
>>> print(base_tool.simplify().latex())
{}^{0}T_{2} = \begin{bmatrix} \cos{\left(q_{1} + q_{2} \right)} & - \sin{\left(q_{1} + q_{2} \right)} & 0 & l_{1} \cos{\left(q_{1} \right)} \\ \sin{\left(q_{1} + q_{2} \right)} & \cos{\left(q_{1} + q_{2} \right)} & 0 & l_{1} \sin{\left(q_{1} \right)} \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}

```

The tip is the point (l2, 0, 0) of `Tool`; the chain resolves it in `Base`:

```python
>>> tip = Tool.point(l2, 0, 0)
>>> (base_tool @ tip).simplify()
Point[Base](l1*cos(q1) + l2*cos(q1 + q2), l1*sin(q1) + l2*sin(q1 + q2), 0)

```

The approach direction is a vector, so the translation drops out:

```python
>>> (base_tool @ Tool.vector(1, 0, 0)).simplify()
Vector[Base](cos(q1 + q2), sin(q1 + q2), 0)

```

## Inverse

The inverse ^B T_A has rotation Rᵀ and translation −Rᵀ t:

```python
>>> base_tool.inverse().translation.simplify()
Vector[Tool](-l1*cos(q2), l1*sin(q2), 0)
>>> (base_tool @ base_tool.inverse()).simplify().entry(1, 4)
0

```
