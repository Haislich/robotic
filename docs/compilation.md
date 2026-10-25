# Compilation

Numbers come only from compiling a quantity whose frame of resolution is already
chosen symbolically. `lambdify` turns any quantity (scalar, point, vector, rotation,
transform) into a JIT-compiled JAX function of a parameters dataclass you write.

## The parameters dataclass

One frozen dataclass names the symbols, usually one for the whole robot. Fields are
matched to symbols by name; order is irrelevant and extra fields are allowed.

```python
>>> from dataclasses import dataclass
>>> from robotic import Frame, Rotation, Scalar, Transform
>>> class Base(Frame, notation="0"):
...     """Fixed frame at the first joint."""
>>> class Link1(Frame, notation="1"):
...     """Frame moving with the first link."""
>>> class Tool(Frame, notation="2"):
...     """Frame moving with the second link."""
>>> @dataclass(frozen=True)
... class Arm:
...     """Joint values and link lengths of a planar 2R arm."""
...     q1: float
...     q2: float
...     l1: float
...     l2: float
>>> q1, q2 = Scalar.symbol("q1"), Scalar.symbol("q2")
>>> l1, l2 = Scalar.symbol("l1"), Scalar.symbol("l2")
>>> base_tool = Transform(target=Base, source=Link1).with_rotation(
...     Rotation.about_z(q1, target=Base, source=Link1)
... ) @ (
...     Transform(target=Link1, source=Tool)
...     .with_rotation(Rotation.about_z(q2, target=Link1, source=Tool))
...     .with_translation(Link1.vector(l1, 0, 0))
... )
>>> tip = (base_tool @ Tool.point(l2, 0, 0)).simplify()
>>> position = tip.lambdify(Arm)
>>> result = position(Arm(q1=0.0, q2=0.0, l1=1.0, l2=0.5))
>>> result
Numeric[Point[Base]]([1.5, 0.0, 0.0])

```

The result keeps the frame in its type, `Numeric[Point[Base]]`, and names it when
printed. Simulation code reads the bare components, a float64 JAX array in the natural
shape: `()` for a scalar, `(3,)` for a point or vector, `(3, 3)` for a rotation,
`(4, 4)` for a transform.

```python
>>> result.components.shape, str(result.components.dtype)
((3,), 'float64')
>>> base_tool.lambdify(Arm)(Arm(q1=0.0, q2=0.0, l1=1.0, l2=0.5)).components.shape
(4, 4)

```

Double precision holds for the whole program: importing `robotic` turns on JAX's
`jax_enable_x64`, so any other JAX code in the same process also defaults to float64.

```python
>>> import jax
>>> jax.config.read("jax_enable_x64")
True

```

Calling with anything but an `Arm`, or building an `Arm` without a field, is a type
error.

## Validation, once, at compile time

Compiling checks the dataclass once: every field is annotated `float` or
`jax.typing.ArrayLike`, and every free symbol has a field. Nothing is checked on each
call.

```python
>>> @dataclass(frozen=True)
... class Shortened:
...     """The arm without its second link length."""
...     q1: float
...     q2: float
...     l1: float
>>> tip.lambdify(Shortened)
Traceback (most recent call last):
...
robotic.symbolic.MissingParameterError: Shortened has no field for l2
>>> @dataclass(frozen=True)
... class Misannotated:
...     """A joint value annotated as text."""
...     q1: str
>>> q1.lambdify(Misannotated)
Traceback (most recent call last):
...
robotic.symbolic.ParameterTypeError: field 'q1' of Misannotated must be annotated float or ArrayLike

```

The annotations are read at runtime, so the annotation's name must exist at runtime:
write `jax.typing.ArrayLike` with `import jax`, not a name imported only under
`TYPE_CHECKING`.

## Batches and gradients

Compiled functions work under `jax.jit`, `jax.vmap` and `jax.grad`. `jax.vmap` is the
only way to batch: every field value is a scalar of shape `()`, which is what `vmap`
hands in. JAX keeps the signature of the function it transforms, so code that batches
annotates its fields and parameters `jax.typing.ArrayLike`, the annotation that lets
`vmap` type-check. Under `vmap` the leading axis of the components is the
configuration and the frame is kept:

```python
>>> import jax
>>> import jax.numpy as jnp
>>> @dataclass(frozen=True)
... class Joint:
...     """One joint value, batchable."""
...     q1: jax.typing.ArrayLike
>>> direction = Base.vector(q1.cos(), q1.sin(), 0).lambdify(Joint)
>>> def direction_at(value: jax.typing.ArrayLike):
...     return direction(Joint(q1=value))
>>> batch = jax.vmap(direction_at)(jnp.array([0.0, jnp.pi / 2]))
>>> batch.components.shape
(2, 3)
>>> batch.components.round(6).tolist()
[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
>>> def x_component(value: jax.typing.ArrayLike):
...     return direction_at(value).components[0]
>>> float(jax.grad(x_component)(0.5))
-0.479425538604203

```

An array given directly, outside `vmap`, is rejected when JAX traces:

```python
>>> direction(Joint(q1=jnp.array([0.0, jnp.pi / 2])))  # doctest: +ELLIPSIS
Traceback (most recent call last):
...
robotic.symbolic.ParameterShapeError: field 'q1' holds an array of shape (2,); every field must be a scalar of shape (), batch configurations with jax.vmap
...

```

## Axis-angle, with its validity flag

The axis-angle solutions are formulas valid where sin θ ≠ 0. Compile them as a whole:
the result carries both solutions and a flag, so the flag cannot be left out by
accident. The components have shape `(2, 4)`: one row per solution, columns
r_x, r_y, r_z, θ. `valid()` exists only on compiled axis-angle solutions, and is true
where |sin θ| exceeds the tolerance (default `1e-9`).

```python
>>> solutions = Rotation.about_z(q1, target=Base, source=Tool).to_axis_angle()
>>> compiled = solutions.lambdify(Joint)
>>> regular = compiled(Joint(q1=0.5))
>>> regular.components.tolist()
[[0.0, 0.0, 1.0, 0.5], [0.0, 0.0, -1.0, -0.5]]
>>> bool(regular.valid())
True
>>> bool(compiled(Joint(q1=0.0)).valid()), bool(compiled(Joint(q1=1e-12)).valid())
(False, False)
>>> bool(compiled(Joint(q1=float(jnp.pi))).valid())
False

```

At θ = π the regular formulas give an axis only by the luck of float rounding, and near
θ = 0 the axis means nothing; the flag reports both. When it is false, the caller
decides: `to_axis_angle` on the exact rotation recognizes the identity and the half
turn exactly.

An axis also compiles alone, like any vector, but then nothing marks the singular
configurations: at q = 0 it is not a number.

```python
>>> axis = solutions.first.axis.lambdify(Joint)
>>> axis(Joint(q1=0.5)).components.tolist()
[0.0, 0.0, 1.0]
>>> axis(Joint(q1=0.0)).components.tolist()
[0.0, 0.0, nan]

```
