# Spec Delta

## Purpose

Models homogeneous transformations between frames, written ^A T_B, so that forward-kinematics chains and the change of frame of points and vectors are checked by the type checker.

## ADDED Requirements

### Requirement: A transform is a rotation and a position between two frames
A transform ^A T_B SHALL carry both frames in its type, SHALL be an element of SE(3) consisting of the rotation ^A R_B and the origin of B expressed in A (a point in A, whose position vector is the fourth column), and SHALL expose its rotation, its origin and the sixteen entries of its 4x4 homogeneous matrix as scalars.

#### Scenario: Matrix layout
- **WHEN** the code builds the homogeneous matrix of the rotation about z by `q1` from `Link1` to `Base` with origin (0, 0, d1) in `Base`
- **THEN** its entries are `[[cos(q1), -sin(q1), 0, 0], [sin(q1), cos(q1), 0, 0], [0, 0, 1, d1], [0, 0, 0, 1]]`

#### Scenario: Reading the origin from the entries
- **WHEN** the code reads row 3, column 4 of the transform with origin (0, 0, d1)
- **THEN** the result is the scalar `d1`

### Requirement: A transform is the homogeneous matrix of a rotation
A transform SHALL be built only as the homogeneous matrix of a rotation, given the origin of the source as a point in the target; without an origin the origins SHALL coincide. A point not in the target, or a vector, given as the origin SHALL be a type error, also when written inline.

#### Scenario: Coincident origins
- **WHEN** the code takes the homogeneous matrix of the identity from `Link1` to `Base` without an origin
- **THEN** the type checker infers a transform from `Link1` to `Base`, and its entries are those of the 4x4 identity

#### Scenario: Origin in the wrong frame, inline
- **WHEN** the code takes the homogeneous matrix of a rotation from `Link1` to `Base` with, inline, a point in `Link1` as origin
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

#### Scenario: A vector is not an origin
- **WHEN** the code takes the homogeneous matrix of a rotation from `Link1` to `Base` with a vector in `Base` as origin
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Transform composition
Composing ^A T_B with ^B T_C SHALL give ^A T_C, and composing transforms whose inner frames differ SHALL be a type error.

#### Scenario: Forward kinematics chain
- **WHEN** the code composes a transform from `Link1` to `Base` with a transform from `Tool` to `Link1`
- **THEN** the type checker infers a transform from `Tool` to `Base`

#### Scenario: Mismatched chain
- **WHEN** the code composes a transform from `Link1` to `Base` with itself
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Transform inverse
The inverse of ^A T_B SHALL be ^B T_A, with rotation Rᵀ and position vector −Rᵀ t, where t is the position vector of B's origin in A.

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
- **AND** executing it raises an error naming the mistake

### Requirement: LaTeX and simplification for transforms
A transform SHALL provide simplification keeping both frames, and a LaTeX rendering of ^A T_B.

#### Scenario: Simplify keeps frames
- **WHEN** the code simplifies a transform from `Tool` to `Base`
- **THEN** the type checker infers a transform from `Tool` to `Base`
