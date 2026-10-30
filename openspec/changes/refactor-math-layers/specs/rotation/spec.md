# Spec Delta

## Purpose

Models rotations between frames, written ^A R_B, so that composing rotations and changing the frame a vector is resolved in are checked by the type checker, and extracts axis-angle pairs symbolically following the course convention.

## ADDED Requirements

### Requirement: A rotation names both frames
A rotation SHALL carry its target frame A and source frame B in its type (^A R_B), and SHALL expose the nine entries of its 3x3 matrix as scalars.

#### Scenario: Elementary rotation
- **WHEN** the code builds the rotation about z by `q1` from `Link1` to `Base`
- **THEN** the type checker infers a symbolic rotation from `Link1` to `Base`, and its entries are `[[cos(q1), -sin(q1), 0], [sin(q1), cos(q1), 0], [0, 0, 1]]`

### Requirement: Entries are counted from 1
A rotation SHALL expose each entry by row and column counted from 1, as the course writes r₁₃, and an index outside 1 to 3 SHALL be a type error.

#### Scenario: Reading r12
- **WHEN** the code reads row 1, column 2 of the rotation about z by `q1`
- **THEN** the result is the scalar `-sin(q1)`

#### Scenario: Index out of range
- **WHEN** the code reads row 0 of a rotation
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Elementary, axis and identity rotations
The library SHALL build the rotation about x, about y and about z by an angle, the rotation about a unit vector by an angle, and the identity, each between two frames.

#### Scenario: Rotation about an axis
- **WHEN** the code builds the rotation about the z unit vector by `q`
- **THEN** its entries equal those of the elementary rotation about z by `q`

#### Scenario: Identity
- **WHEN** the code builds the identity rotation
- **THEN** its entries are those of the 3x3 identity

### Requirement: Frames are fixed by the receiver
No operation that builds a rotation from two arguments SHALL infer a frame from both; the frames SHALL be fixed by the receiver or by explicit frame arguments, and an argument in another frame SHALL be a type error even when written inline.

#### Scenario: Inline argument in the wrong frame
- **WHEN** the code passes, inline, a value in `Link1` where the receiver requires `Base`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Composition
Composing ^A R_B with ^B R_C SHALL give ^A R_C, and composing rotations whose inner frames differ SHALL be a type error.

#### Scenario: Chaining
- **WHEN** the code composes a rotation from `Link1` to `Base` with a rotation from `Tool` to `Link1`
- **THEN** the type checker infers a rotation from `Tool` to `Base`

#### Scenario: Mismatched chain
- **WHEN** the code composes a rotation from `Link1` to `Base` with itself
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

#### Scenario: Skipping an intermediate frame
- **WHEN** the frames `Prime` and `Second` lie between `Base` and `Tool`, and the code composes the rotation from `Prime` to `Base` with the rotation from `Tool` to `Second`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Inverse
The inverse of ^A R_B SHALL be ^B R_A, whose entries are the transposed entries.

#### Scenario: Inverse composes to identity
- **WHEN** the code composes a rotation with its inverse and simplifies
- **THEN** its entries are those of the 3x3 identity

### Requirement: Changing the frame of resolution
Applying ^A R_B to a vector resolved in B SHALL give the same arrow resolved in A, and applying it to a vector in another frame SHALL be a type error.

#### Scenario: Resolving in the target frame
- **WHEN** the rotation about z by π/2 from `Tool` to `Base` is applied to the x unit vector of `Tool`
- **THEN** the result is the vector (0, 1, 0) in `Base`

#### Scenario: Wrong source frame
- **WHEN** a rotation from `Tool` to `Base` is applied to a vector in `Base`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: A rotation does not act on points
Applying a rotation to a point SHALL be a type error, because a rotation relates axes and says nothing about where the origins are.

#### Scenario: Rotation applied to a point
- **WHEN** a rotation from `Link1` to `Base` is applied to a point in `Link1`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Axis-angle extraction is symbolic
Extracting axis-angle from ^A R_B SHALL return, without evaluating any symbol, two axis-angle solutions with the angle in (−π, π]; each axis SHALL be a unit vector satisfying R r = r, so it has the same components in A and B.

#### Scenario: Regular case
- **WHEN** the code extracts axis-angle from the rotation about z by π/3
- **THEN** the first solution is axis (0, 0, 1) with angle π/3 and the second is axis (0, 0, −1) with angle −π/3

### Requirement: Constant rotations are resolved, symbolic ones get the regular formulas
A constant rotation SHALL be resolved as the regular, half-turn or identity case, deciding on its values with the default tolerance; a symbolic rotation SHALL get the regular formulas, with no condition attached, and SHALL never raise because of them. Where those formulas break, the axis is undefined (open on purpose).

#### Scenario: Symbolic rotation
- **WHEN** the code extracts axis-angle from the rotation about z by a symbol `q`
- **THEN** both solutions are the regular formulas as expressions in `q`, no value is substituted, and no condition on `q` is attached

### Requirement: Half turn
For a rotation by π, the two solutions SHALL be (r, π) and (−r, π).

#### Scenario: Half turn about x
- **WHEN** the code extracts axis-angle from the rotation about x by π
- **THEN** the solutions are axis (1, 0, 0) with angle π and axis (−1, 0, 0) with angle π

#### Scenario: Symbolic product that is the identity
- **WHEN** the code extracts axis-angle from `R @ R.inverse()`, where `R` is the rotation about x by a symbol `q`
- **THEN** the product is a symbolic rotation, the regular formulas apply, and no error is raised

#### Scenario: Evaluating first gives the exact case
- **WHEN** the code evaluates `R @ R.inverse()` with `q` set to 1 and extracts axis-angle
- **THEN** the solutions are the free axis (0, 0, 1) with angle 0 and (0, 0, −1) with angle 0

#### Scenario: Half turn about a symbolic axis
- **WHEN** the code extracts axis-angle from the rotation by π about the normalized vector (a, 1, 0)
- **THEN** the regular formulas apply, because the rotation is symbolic, and no error is raised, although its skew part vanishes for every `a`

#### Scenario: Constant half turn about z
- **WHEN** the code extracts axis-angle from the constant rotation about z by π
- **THEN** the solutions are axis (0, 0, 1) with angle π and axis (0, 0, −1) with angle π

### Requirement: Identity gives a free axis
For the identity rotation, the two solutions SHALL be (a, 0) and (−a, 0), where a is a free axis that is a unit vector, z unless the caller chooses another.

#### Scenario: Default free axis
- **WHEN** the code extracts axis-angle from the identity
- **THEN** the solutions are axis (0, 0, 1) with angle 0 and axis (0, 0, −1) with angle 0, and the axis is marked free

#### Scenario: Chosen free axis
- **WHEN** the code extracts axis-angle from the identity with free axis x
- **THEN** the solutions are axis (1, 0, 0) with angle 0 and axis (−1, 0, 0) with angle 0

### Requirement: LaTeX and simplification for rotations
A rotation SHALL provide simplification keeping both frames, and a LaTeX rendering of ^A R_B.

#### Scenario: Simplify keeps frames
- **WHEN** the code simplifies a rotation from `Tool` to `Base`
- **THEN** the type checker infers a rotation from `Tool` to `Base`
