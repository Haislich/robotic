# Transforms

^A T_B is the pose of frame B relative to frame A: the rotation ^A R_B and the origin
of B resolved in A, the homogeneous matrix [R | t; 0 0 0 1]. On a point it is
R p + t; on a vector it is R v, because a translation does not move an arrow.

## A planar 2R arm, end to end

Two revolute joints about z, links of length l1 and l2. Frame `Link1` turns by q1 at
the base; frame `Tool` turns by q2 at the end of the first link.

```python
>>> from robotic import Frame, Rotation, Scalar, Transform
>>> class Base(Frame, latex="0"):
...     """Fixed frame at the first joint."""
>>> class Link1(Frame, latex="1"):
...     """Frame moving with the first link."""
>>> class Tool(Frame, latex="2"):
...     """Frame moving with the second link."""
>>> q1, q2 = Scalar.symbol("q1"), Scalar.symbol("q2")
>>> l1, l2 = Scalar.symbol("l1"), Scalar.symbol("l2")

```

A transform is built from its rotation and the origin's three coordinates, read in
the rotation's reference frame, like the exam's [R | t]:

```python
>>> base_link1 = Transform(Rotation.about_z(q1, of=Link1, relative_to=Base), 0, 0, 0)

```

When the origin is already a point, the rotation takes it; a point in the wrong frame
is a type error:

```python
>>> link1_tool = Rotation.about_z(q2, of=Tool, relative_to=Link1).with_origin(
...     Link1.point(l1, 0, 0)
... )
>>> link1_tool.origin
Point[Link1](l1, 0, 0)

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

The inverse ^B T_A has rotation Rᵀ and origin −Rᵀ t:

```python
>>> base_tool.inverse().origin.simplify()
Point[Tool](-l1*cos(q2), l1*sin(q2), 0)
>>> (base_tool @ base_tool.inverse()).simplify().entry(1, 4)
0

```
