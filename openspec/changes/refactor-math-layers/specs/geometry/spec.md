# Spec Delta

## Purpose

Models frames, and points and vectors resolved in a frame, so that the frame a set of components is read in is part of the value's type and affine mistakes are rejected before the code runs.

## ADDED Requirements

### Requirement: Frames are user-declared types
A frame SHALL be a type the user declares by subclassing the library's frame type, and every point and vector SHALL belong to exactly one frame, fixed by the type.

#### Scenario: Declaring a frame
- **WHEN** the user declares `class Tool(Frame)` and creates a vector with components (1, 0, 0) from `Tool`
- **THEN** the type checker infers a vector in `Tool`

### Requirement: Frames give points and vectors
A frame SHALL create a point from three numbers (origin and axes), a vector from three numbers (axes only), and its own origin as a point.

#### Scenario: Point, vector and origin
- **WHEN** the code creates a point, a vector and the origin from `Tool`
- **THEN** the type checker infers a point in `Tool`, a vector in `Tool` and a point in `Tool`

### Requirement: Components are readable
A point or vector SHALL expose its x, y and z components as scalars and its frame at runtime.

#### Scenario: Reading components and frame
- **WHEN** `tip` is the point (l2, 0, 0) in `Tool`
- **THEN** its x component is the scalar `l2` and its frame is `Tool`

### Requirement: Affine arithmetic
Within one frame, point minus point SHALL give a vector, point plus or minus vector SHALL give a point, vector plus or minus vector SHALL give a vector, and a number times or divided into a vector SHALL give a vector.

#### Scenario: Coordinates as a vector
- **WHEN** `tip` is a point in `Tool` and the code computes `tip - Tool.origin()`
- **THEN** the type checker infers a vector in `Tool` with the same components as `tip`

#### Scenario: Displacing a point
- **WHEN** the code adds a vector in `Tool` to a point in `Tool`
- **THEN** the type checker infers a point in `Tool`

### Requirement: Affine mistakes are type errors
Adding two points, scaling a point and negating a point SHALL be type errors, and SHALL raise when executed.

#### Scenario: Point plus point
- **WHEN** the code computes `tip + tip` for a point `tip`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

#### Scenario: Scaling a point
- **WHEN** the code computes `2 * tip` for a point `tip`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Mixing frames is a type error
Every binary operation between points and vectors SHALL require both operands in the same frame, and assigning a value in one frame to a variable annotated with another frame SHALL be a type error. Every such operation SHALL also raise at runtime when the frames stored in the operands differ, whatever their annotations.

#### Scenario: Vectors in different frames
- **WHEN** the code adds a vector in `Tool` to a vector in `Base`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

#### Scenario: Widening a bound value
- **WHEN** `v` is bound to a vector in `Tool` and then assigned to a variable annotated as a vector in `Base`
- **THEN** the type checker reports an error
- **AND** executing the assignment does not raise (an annotation is not checked at runtime), and the first operation between it and a vector in `Base` raises an error naming both frames

#### Scenario: A value in the wrong frame keeps its own frame
- **WHEN** `v1: Vector[B2] = Vector(0, 0, 1, B1)`, `v2 = Vector(0, 0, 1, B2)`, and the code computes `v1 + v2`, `v2 + v1` and `v1 + v1`
- **THEN** the type checker reports the assignment of `v1`
- **AND** `v1 + v2` and `v2 + v1` raise an error naming `B1` and `B2`, and `v1 + v1` is the vector (0, 0, 2) in `B1`

### Requirement: Vector algebra
A vector SHALL provide the dot product and the cross product with a vector in the same frame, its norm as a scalar, and normalization to a unit vector in the same frame.

#### Scenario: Cross product of axes
- **WHEN** the code computes the cross product of the x and y unit vectors of `Tool`
- **THEN** the result is the z unit vector of `Tool`

#### Scenario: Norm
- **WHEN** the code computes the norm of the vector (3, 4, 0) in `Tool`
- **THEN** the result is the scalar `5`

### Requirement: Unit vectors come only from normalization
Every unit vector SHALL be the normalization of a vector: building a unit vector from a vector SHALL divide its components by their simplified norm, and no way of building a unit vector SHALL accept components without normalizing them.

#### Scenario: Building a unit vector normalizes
- **WHEN** the code builds a unit vector from the vector (3, 4, 0) in `Tool`
- **THEN** the result is the unit vector (3/5, 4/5, 0) in `Tool`

#### Scenario: Simplified norm
- **WHEN** the code normalizes the vector (cos(q), sin(q), 0)
- **THEN** the result is (cos(q), sin(q), 0), with no square root left in the components

### Requirement: Normalizing a zero vector raises
Normalizing a vector whose components simplify to zero SHALL raise an error, because no direction exists.

#### Scenario: Zero vector
- **WHEN** the code normalizes the vector (0, 0, 0)
- **THEN** an error is raised

### Requirement: Simplification and LaTeX for points and vectors
A point or vector SHALL provide simplification returning the same type in the same frame, and a LaTeX rendering that names its frame by the notation declared with the frame, or by the frame's class name when none is declared.

#### Scenario: Simplify keeps the frame
- **WHEN** the code simplifies a vector in `Base`
- **THEN** the type checker infers a vector in `Base`

#### Scenario: Default name
- **WHEN** `Tool` is declared without a notation and the code renders a vector in `Tool` as LaTeX
- **THEN** the string contains `{}^{Tool}`

#### Scenario: Declared notation
- **WHEN** `Base` is declared with the notation `0` and the code renders a vector in `Base` as LaTeX
- **THEN** the string contains `{}^{0}`

#### Scenario: Misspelled declaration
- **WHEN** a frame is declared with a misspelled notation keyword
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake
