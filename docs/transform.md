# Transforms

^A T_B maps positions expressed in the source B to the target A. It is an element of
SE(3), the homogeneous matrix [R p; 0 0 0 1] of the rotation ^A R_B, whose fourth
column is the translation: the position of B's origin in A. On a position it is R p + t; on a
displacement it is R v, because a translation does not move a displacement. Like a
rotation, it is constant only when its rotation and its translation both are.

## A planar 2R arm, end to end

Two revolute joints about z, links of length l1 and l2. Frame `Link1` turns by q1 at
the base; frame `Tool` turns by q2 at the end of the first link.

```python
>>> from robotic import Frame, Rotation, Symbol
>>> class Base(Frame, notation="0"):
...     """Fixed frame at the first joint."""
>>> class Link1(Frame, notation="1"):
...     """Frame moving with the first link."""
>>> class Tool(Frame, notation="2"):
...     """Frame moving with the second link."""
>>> q1, q2 = Symbol("q1"), Symbol("q2")
>>> l1, l2 = Symbol("l1"), Symbol("l2")

```

A transform is the homogeneous matrix of its rotation, given the translation: the
origin of the source as a position in the target; without a translation, the two
origins coincide. The rotation fixes both frames, so a translation in another frame is
a type error, and so is a displacement, whose homogeneous form [v; 0] would leave a
matrix outside SE(3):

```python
>>> base_link1 = Rotation.about_z(q1, target=Base, source=Link1).homogeneous()
>>> link1_tool = Rotation.about_z(q2, target=Link1, source=Tool).homogeneous(
...     Link1.position(l1, 0, 0)
... )
>>> link1_tool.translation
Position[Link1](l1, 0, 0)

```

Composition chains the inner frames, ^0 T_1 ^1 T_2 = ^0 T_2:

```python
>>> base_tool = base_link1 @ link1_tool
>>> base_tool.entry(1, 4)
l1*cos(q1)
>>> print(base_tool.simplify().latex())
{}^{0}T_{2} = \begin{bmatrix} \cos{\left(q_{1} + q_{2} \right)} & - \sin{\left(q_{1} + q_{2} \right)} & 0 & l_{1} \cos{\left(q_{1} \right)} \\ \sin{\left(q_{1} + q_{2} \right)} & \cos{\left(q_{1} + q_{2} \right)} & 0 & l_{1} \sin{\left(q_{1} \right)} \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}

```

The tip is the position (l2, 0, 0) in `Tool`; the chain resolves it in `Base`:

```python
>>> tip = Tool.position(l2, 0, 0)
>>> (base_tool @ tip).simplify()
Position[Base](l1*cos(q1) + l2*cos(q1 + q2), l1*sin(q1) + l2*sin(q1 + q2), 0)

```

The approach direction is a displacement, so the translation drops out:

```python
>>> (base_tool @ Tool.displacement(1, 0, 0)).simplify()
Displacement[Base](cos(q1 + q2), sin(q1 + q2), 0)

```

## Inverse

The inverse ^B T_A has rotation Rᵀ, and its translation is the origin of A resolved
in B:

```python
>>> base_tool.inverse().translation.simplify()
Position[Tool](-l1*cos(q2), l1*sin(q2), 0)
>>> (base_tool @ base_tool.inverse()).simplify().entry(1, 4)
0

```
