# Spec Delta

## Purpose

Models homogeneous transformations between frames, written ^A T_B, so that forward-kinematics chains and the change of frame of points and vectors are checked by the type checker.

## ADDED Requirements

### Requirement: A transform is a rotation and an origin
A transform ^A T_B SHALL consist of the rotation ^A R_B and the origin of B as a point in A, SHALL carry both frames in its type, and SHALL expose its rotation, its origin and the sixteen entries of its 4x4 homogeneous matrix as scalars.

#### Scenario: Matrix layout
- **WHEN** the code builds the transform from `Link1` to `Base` with rotation about z by `q1` and origin (0, 0, d1) in `Base`
- **THEN** its entries are `[[cos(q1), -sin(q1), 0, 0], [sin(q1), cos(q1), 0, 0], [0, 0, 1, d1], [0, 0, 0, 1]]`

#### Scenario: Reading the translation from the entries
- **WHEN** the code reads row 3, column 4 of the transform with origin (0, 0, d1)
- **THEN** the result is the scalar `d1`

### Requirement: The origin is checked against the rotation
Building a transform SHALL fix both frames from the rotation, and an origin that is not a point in the rotation's target frame SHALL be a type error, also when written inline.

#### Scenario: Building from a rotation and three numbers
- **WHEN** the code builds a transform from the rotation from `Link1` to `Base` and the numbers (0, 0, d1)
- **THEN** the type checker infers a transform from `Link1` to `Base`, and its origin is the point (0, 0, d1) in `Base`

#### Scenario: Origin in the wrong frame, inline
- **WHEN** the code builds a transform from the rotation from `Link1` to `Base` and, inline, the origin of `Link1`
- **THEN** the type checker reports an error

### Requirement: Transform composition
Composing ^A T_B with ^B T_C SHALL give ^A T_C, and composing transforms whose inner frames differ SHALL be a type error.

#### Scenario: Forward kinematics chain
- **WHEN** the code composes a transform from `Link1` to `Base` with a transform from `Tool` to `Link1`
- **THEN** the type checker infers a transform from `Tool` to `Base`

#### Scenario: Mismatched chain
- **WHEN** the code composes a transform from `Link1` to `Base` with itself
- **THEN** the type checker reports an error

### Requirement: Transform inverse
The inverse of ^A T_B SHALL be ^B T_A, with rotation Rᵀ and origin −Rᵀ t.

#### Scenario: Inverse composes to identity
- **WHEN** the code composes a transform with its inverse and simplifies
- **THEN** its entries are those of the 4x4 identity

### Requirement: Transforms act on points with translation
Applying ^A T_B to a point in B SHALL give R p + t as a point in A.

#### Scenario: Tip of a planar 2R arm
- **WHEN** the chain from `Tool` to `Base` of a planar 2R arm (joint angles q1, q2, link lengths l1, l2) is applied to the point (l2, 0, 0) in `Tool` and simplified
- **THEN** the result is the point (l1 cos(q1) + l2 cos(q1 + q2), l1 sin(q1) + l2 sin(q1 + q2), 0) in `Base`

### Requirement: Transforms act on vectors without translation
Applying ^A T_B to a vector in B SHALL give R v as a vector in A.

#### Scenario: Translation does not move a vector
- **WHEN** a transform with identity rotation and origin (5, 0, 0) is applied to the vector (1, 0, 0)
- **THEN** the result is the vector (1, 0, 0)

### Requirement: Wrong source frame is a type error
Applying a transform to a point or vector not in its source frame SHALL be a type error.

#### Scenario: Point in the wrong frame
- **WHEN** a transform from `Link1` to `Base` is applied to a point in `Tool`
- **THEN** the type checker reports an error

### Requirement: LaTeX and simplification for transforms
A transform SHALL provide simplification keeping both frames, and a LaTeX rendering of ^A T_B.

#### Scenario: Simplify keeps frames
- **WHEN** the code simplifies a transform from `Tool` to `Base`
- **THEN** the type checker infers a transform from `Tool` to `Base`
