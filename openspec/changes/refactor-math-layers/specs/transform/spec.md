# Spec Delta

## Purpose

Models homogeneous transformations between frames, written ^A T_B, so that forward-kinematics chains and the change of frame of points and vectors are checked by the type checker.

## ADDED Requirements

### Requirement: A transform is a rotation and a translation between two frames
A transform ^A T_B SHALL carry both frames in its type, SHALL consist of the rotation ^A R_B and the translation ^A p (the origin of B expressed in A, a vector in A), and SHALL expose its rotation, its translation and the sixteen entries of its 4x4 homogeneous matrix as scalars.

#### Scenario: Matrix layout
- **WHEN** the code builds the transform from `Link1` to `Base` with rotation about z by `q1` and translation (0, 0, d1) in `Base`
- **THEN** its entries are `[[cos(q1), -sin(q1), 0, 0], [sin(q1), cos(q1), 0, 0], [0, 0, 1, d1], [0, 0, 0, 1]]`

#### Scenario: Reading the translation from the entries
- **WHEN** the code reads row 3, column 4 of the transform with translation (0, 0, d1)
- **THEN** the result is the scalar `d1`

### Requirement: A transform is built from its frames
Building a transform SHALL start from its target and source frames, giving the identity between them; setting its rotation or translation SHALL return a new transform with the same frames, and a rotation not from the source to the target, or a translation not in the target, SHALL be a type error, also when written inline.

#### Scenario: Identity between two frames
- **WHEN** the code builds the transform from `Link1` to `Base` without setting any part
- **THEN** the type checker infers a transform from `Link1` to `Base`, and its entries are those of the 4x4 identity

#### Scenario: Setting a part returns a new transform
- **WHEN** the code sets the translation of a transform `t` and binds the result to `u`
- **THEN** `t` is unchanged and `u` has the new translation and the same frames

#### Scenario: Rotation in the wrong frames, inline
- **WHEN** the code sets, inline, a rotation from `Tool` to `Base` on the transform from `Link1` to `Base`
- **THEN** the type checker reports an error

#### Scenario: Translation in the wrong frame, inline
- **WHEN** the code sets, inline, a vector in `Link1` as the translation of the transform from `Link1` to `Base`
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
The inverse of ^A T_B SHALL be ^B T_A, with rotation Rᵀ and translation −Rᵀ t.

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
- **WHEN** a transform with identity rotation and translation (5, 0, 0) is applied to the vector (1, 0, 0)
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
