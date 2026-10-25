# Spec Delta

## Purpose

Turns a symbolic quantity into a fast JAX function of a user-written parameters dataclass, so that numbers come only from evaluating a resolution already chosen symbolically, while the result keeps the meaning (frame) in its type.

## ADDED Requirements

### Requirement: Compiling against a parameters dataclass
Every quantity (scalar, point, vector, rotation, transform) SHALL compile, given a dataclass type whose fields name the symbols, to a callable that takes one instance of that dataclass; fields are matched to symbols by name, their order is irrelevant, and fields without a matching symbol are allowed.

#### Scenario: One parameters class for the whole robot
- **WHEN** the tip of a planar 2R arm (symbols q1, q2, l1, l2) is compiled against a dataclass with fields q1, q2, l1, l2 and a fifth unused field
- **THEN** compilation succeeds and calling it with an instance returns the tip's coordinates

#### Scenario: Wrong argument type
- **WHEN** a compiled function is called with a bare float instead of a dataclass instance
- **THEN** the type checker reports an error

### Requirement: Compilation validates once, at compile time
Compiling SHALL raise when a free symbol of the quantity has no field of the same name, or when a field is not annotated `float` (string annotations resolved), and SHALL perform no such check on each call.

#### Scenario: Missing field
- **WHEN** a quantity containing `l2` is compiled against a dataclass without an `l2` field
- **THEN** an error naming `l2` is raised at compile time

#### Scenario: Field not a float
- **WHEN** a dataclass declares a field `q1: str`
- **THEN** compiling against it raises an error naming `q1`

#### Scenario: String annotation
- **WHEN** a dataclass declares `q1: "float"`
- **THEN** compiling against it succeeds

### Requirement: Results have the natural shape
Calling a compiled function SHALL return JAX arrays in double precision with shape `()` for a scalar, `(3,)` for a point or vector, `(3, 3)` for a rotation and `(4, 4)` for a transform.

#### Scenario: Vector shape
- **WHEN** a compiled vector is called
- **THEN** its components have shape `(3,)` and dtype float64

### Requirement: Results keep the frame
The result of calling a compiled point, vector, rotation or transform SHALL keep its frame(s) in its type and name them in its printed form, and SHALL expose its components as a JAX array; it SHALL offer no operation that changes the frame.

#### Scenario: Compiled point in Base
- **WHEN** a point in `Base` is compiled and called
- **THEN** the type checker infers a result tied to a point in `Base`, its printed form names `Base`, and its components are a JAX array

### Requirement: Compiled functions work with JAX transformations
A compiled function and its result SHALL work under `jax.jit`, `jax.vmap` and `jax.grad`; under `vmap`, leading axes of the components are configurations and the frame is kept.

#### Scenario: Batched configurations
- **WHEN** a compiled vector is mapped with `vmap` over 2 configurations
- **THEN** the components have shape `(2, 3)` and the frame is unchanged

#### Scenario: Gradient
- **WHEN** `grad` is taken of the x component of the compiled vector (sin(q), 0, 0) with respect to `q` at `q = 0.5`
- **THEN** the result is cos(0.5) = 0.8775825618903728

### Requirement: Axis-angle solutions compile with their validity flag
Compiling axis-angle solutions SHALL return both solutions together with a boolean flag that is true only where the validity condition holds by more than the tolerance (default 1e-9), and a validity condition SHALL compile on its own to such a flag.

#### Scenario: Regular configuration
- **WHEN** the axis-angle solutions of the rotation about z by `q` are compiled and called at `q = 0.5`
- **THEN** the flag is true and the first axis is (0, 0, 1)

#### Scenario: Identity
- **WHEN** they are called at `q = 0`
- **THEN** the flag is false

#### Scenario: Half turn
- **WHEN** they are called at `q = π`
- **THEN** the flag is false

### Requirement: Compiled values match the symbolic ones
For any parameter values, the compiled result SHALL equal the symbolic quantity with those values substituted, within floating-point tolerance.

#### Scenario: Oracle comparison
- **WHEN** a compiled rotation is called at sampled joint values, including values near singular configurations
- **THEN** the matrix matches the roboticstoolbox/spatialmath oracle within tolerance
