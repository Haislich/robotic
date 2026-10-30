# Compilation

Numbers come only from compiling a quantity whose frame of resolution is already
chosen symbolically. `compile` turns any symbolic quantity (expression, position,
displacement, rotation, transform, axis-angle solutions) into a JIT-compiled JAX
function of a parameters dataclass you write; a constant compiles with no parameters,
straight to its value. The expressions are compiled as written, without simplifying them; repeated
subexpressions are computed once, so long chains compile quickly.

## The parameters dataclass

One frozen dataclass names the symbols, usually one for the whole robot. Fields are
matched to symbols by name; order is irrelevant and extra fields are allowed.

```python
>>> from dataclasses import dataclass
>>> from robotic import Frame, Rotation, Scalar, Symbol
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
>>> q1, q2 = Symbol("q1"), Symbol("q2")
>>> l1, l2 = Symbol("l1"), Symbol("l2")
>>> base_tool = Rotation.about_z(q1, target=Base, source=Link1).homogeneous() @ (
...     Rotation.about_z(q2, target=Link1, source=Tool).homogeneous(
...         Link1.position(l1, 0, 0)
...     )
... )
>>> tip = (base_tool @ Tool.position(l2, 0, 0)).simplify()
>>> position = tip.compile(Arm)
>>> result = position(Arm(q1=0.0, q2=0.0, l1=1.0, l2=0.5))
>>> result
Numeric[Position[Base]]([1.5, 0.0, 0.0])

```

The result keeps the frame in its type, `Numeric[Position[Base]]`, and names it when
printed. Simulation code reads the bare components, a float64 JAX array in the natural
shape: `()` for a scalar, `(3,)` for a position or displacement, `(3, 3)` for a
rotation, `(4, 4)` for a transform. Floats live here, on the numeric side; the
symbolic side takes exact numbers only. Compiled values have no `==`: compare their
components with a tolerance, e.g. `jnp.allclose(a.components, b.components)`.

```python
>>> result.components.shape, str(result.components.dtype)
((3,), 'float64')
>>> base_tool.compile(Arm)(Arm(q1=0.0, q2=0.0, l1=1.0, l2=0.5)).components.shape
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
error, and raises when executed.

## Constants

A constant has nothing to vary, so it compiles with no parameters and gives its value
directly. The value is the nearest double; the exact one stays symbolic.

```python
>>> Base.position(1, 2, Scalar(2).sqrt()).compile()
Numeric[Position[Base]]([1.0, 2.0, 1.4142135623730951])

```

A symbolic quantity compiled without parameters is a type error, and raises when
executed.

## Validation, once, at compile time

Compiling checks the dataclass once: every field is annotated `float` or
`jax.typing.ArrayLike`, and every free symbol has a field. Each call only checks that
it was given an instance of the dataclass.

```python
>>> @dataclass(frozen=True)
... class Shortened:
...     """The arm without its second link length."""
...     q1: float
...     q2: float
...     l1: float
>>> tip.compile(Shortened)
Traceback (most recent call last):
...
robotic.symbolic.MissingParameterError: Shortened has no field for l2
>>> @dataclass(frozen=True)
... class Misannotated:
...     """A joint value annotated as text."""
...     q1: str
>>> q1.compile(Misannotated)
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
>>> direction = Base.displacement(q1.cos(), q1.sin(), 0).compile(Joint)
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

Derivatives are taken of `.components`, the bare numbers. The derivative of a rotation
matrix is not a rotation, so it does not come back as a `Numeric[Rotation[...]]`:

```python
>>> turn = Rotation.about_z(q1, target=Base, source=Link1).compile(Joint)
>>> derivative = jax.jacrev(lambda value: turn(Joint(q1=value)).components)(0.0)
>>> derivative.shape, float(derivative[1, 0]), float(derivative[0, 1])
((3, 3), 1.0, -1.0)

```

An array given directly, outside `vmap`, is rejected when JAX traces:

```python
>>> direction(Joint(q1=jnp.array([0.0, jnp.pi / 2])))  # doctest: +ELLIPSIS
Traceback (most recent call last):
...
robotic.symbolic.ParameterShapeError: field 'q1' holds an array of shape (2,); every field must be a scalar of shape (), batch configurations with jax.vmap
...

```

## Axis-angle

The axis-angle solutions compile as a whole: the components have shape `(2, 4)`, one
row per solution, columns r_x, r_y, r_z, θ.

```python
>>> solutions = Rotation.about_z(q1, target=Base, source=Tool).to_axis_angle()
>>> compiled = solutions.compile(Joint)
>>> compiled(Joint(q1=0.5)).components.tolist()
[[0.0, 0.0, 1.0, 0.5], [0.0, 0.0, -1.0, -0.5]]

```

For a symbolic rotation these are the regular formulas, which break where sin θ = 0
(θ = 0 and θ = π). What a compiled function returns there is not decided yet: the
library promises no value at singular configurations. When exact values are known,
evaluate the rotation first: `to_axis_angle` on the constant rotation recognizes the
identity and the half turn.
