# compilation Specification

## Purpose

Turns a symbolic quantity into a fast JAX function of a user-written parameters dataclass, so that numbers come only from evaluating a resolution already chosen symbolically, while the result keeps the meaning (frame) in its type.

## Requirements

### Requirement: Compiling against a parameters dataclass
Every symbolic quantity (expression, point, vector, rotation, transform, axis-angle result) SHALL compile, given a dataclass type whose fields name the symbols, to a callable that takes one instance of that dataclass; fields are matched to symbols by name, their order is irrelevant, and fields without a matching symbol are allowed.

#### Scenario: An instance instead of the class
- **WHEN** a quantity is compiled against an instance of the parameters dataclass instead of the class
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Constants compile without parameters
Every constant quantity SHALL compile with no argument directly to its compiled result; compiling a symbolic quantity with no argument SHALL be a type error and SHALL raise.

#### Scenario: Constant position
- **WHEN** the constant position (1, 2, 2) in `Base` is compiled with no argument
- **THEN** the result is the compiled position with components (1.0, 2.0, 2.0) in `Base`

#### Scenario: Symbolic quantity without parameters
- **WHEN** the rotation about z by a symbol `q` is compiled with no argument
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

#### Scenario: One parameters class for the whole robot
- **WHEN** the tip of a planar 2R arm (symbols q1, q2, l1, l2) is compiled against a dataclass with fields q1, q2, l1, l2 and a fifth unused field
- **THEN** compilation succeeds and calling it with an instance returns the tip's coordinates

#### Scenario: Wrong argument type
- **WHEN** a compiled function is called with a bare float instead of a dataclass instance
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Compilation validates once, at compile time
Compiling SHALL raise when a free symbol of the quantity has no field of the same name, or when a field is annotated neither `float` nor `jax.typing.ArrayLike` (string annotations resolved), and SHALL perform no such check on each call.

#### Scenario: Missing field
- **WHEN** a quantity containing `l2` is compiled against a dataclass without an `l2` field
- **THEN** an error naming `l2` is raised at compile time

#### Scenario: Field not a float
- **WHEN** a dataclass declares a field `q1: str`
- **THEN** compiling against it raises an error naming `q1`

#### Scenario: String annotation
- **WHEN** a dataclass declares `q1: "float"`
- **THEN** compiling against it succeeds

#### Scenario: Batchable field
- **WHEN** a dataclass declares `q1: ArrayLike` and a function of `q: ArrayLike` calls the compiled function with it under `jax.vmap`
- **THEN** compiling succeeds and the type checker reports no error

### Requirement: Results have the natural shape
Calling a compiled function SHALL return JAX arrays in double precision with shape `()` for a scalar, `(3,)` for a point or vector, `(3, 3)` for a rotation, `(4, 4)` for a transform and `(2, 4)` for axis-angle solutions (rows the two solutions, columns the axis components and the angle).

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

### Requirement: Batching is through vmap
Every field value SHALL be a scalar of shape `()`; a field holding an array SHALL raise an error that names the field and points to `jax.vmap`, checked when JAX traces, not on each call.

#### Scenario: Array passed directly
- **WHEN** a compiled rotation is called with a dataclass whose `ArrayLike` field holds an array of shape `(2,)`
- **THEN** an error naming the field is raised

#### Scenario: Array batched with vmap
- **WHEN** the same call is mapped with `jax.vmap` over that array
- **THEN** the components have shape `(2, 3, 3)`

#### Scenario: Gradient
- **WHEN** `grad` is taken of the x component of the compiled vector (sin(q), 0, 0) with respect to `q` at `q = 0.5`
- **THEN** the result is cos(0.5) = 0.8775825618903728

### Requirement: Axis-angle solutions compile
Compiling axis-angle solutions SHALL return both solutions in one result; this change promises no value at singular configurations (open on purpose, see the proposal's Out of Scope).

#### Scenario: Regular configuration
- **WHEN** the axis-angle solutions of the rotation about z by `q` are compiled and called at `q = 0.5`
- **THEN** the first axis is (0, 0, 1) with angle 0.5 and the second is (0, 0, −1) with angle −0.5

### Requirement: Compiled results have no equality
Comparing compiled results with `==` or `!=`, hashing them and using them as a truth value SHALL raise an error pointing to a tolerance comparison of `.components`.

#### Scenario: Comparing two compiled results
- **WHEN** the code compares two compiled points with `==`
- **THEN** it raises an error pointing to `.components` and a tolerance comparison

### Requirement: Compiled values match the symbolic ones
For any parameter values, the compiled result SHALL equal the symbolic quantity with those values substituted, within floating-point tolerance.

#### Scenario: Oracle comparison
- **WHEN** a compiled rotation is called at sampled joint values, including values near singular configurations
- **THEN** the matrix matches the roboticstoolbox/spatialmath oracle within tolerance
