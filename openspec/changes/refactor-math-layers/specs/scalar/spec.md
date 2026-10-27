# Spec Delta

## Purpose

Provides the typed scalar every other layer is built on: a symbolic expression whose type is known to the type checker, so exam-style formulas stay typed from the first symbol to the last simplification.

## ADDED Requirements

### Requirement: Typed scalar arithmetic
A scalar SHALL support addition, subtraction, multiplication, division, power and negation with another scalar or an `int`, on either side, and every result SHALL be typed as a scalar.

#### Scenario: Mixed arithmetic stays typed
- **WHEN** `q` is a symbol and the code computes `q + 1`, `1 + q`, `q * q`, `2 * q - 0.5` and `-q`
- **THEN** the type checker infers a scalar for each result

#### Scenario: Non-numbers are rejected
- **WHEN** the code computes `q + "a"`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Symbols are created only by the library
The library SHALL provide the only way to create a symbol, from a name, with fixed assumptions, so that two symbols with the same name are the same symbol.

#### Scenario: Same name, same symbol
- **WHEN** `q2` is created twice by name and the two are added
- **THEN** the result is `2*q2` with one free symbol

### Requirement: Exact constants
The library SHALL provide exact π, square root and exponential as scalar operations, so exact values never pass through a float.

#### Scenario: Exact pi
- **WHEN** the code computes π divided by 2 through the library
- **THEN** the result prints as `pi/2` and contains no float

#### Scenario: Exact square root
- **WHEN** the code takes the square root of the scalar 2
- **THEN** the result prints as `sqrt(2)`

### Requirement: Elementary functions as methods
A scalar SHALL provide cosine, sine, tangent, two-argument arctangent, square root and exponential as methods returning a scalar.

#### Scenario: Trigonometry
- **WHEN** `q` is a symbol and the code calls its cosine method
- **THEN** the result is a scalar printing as `cos(q)`

### Requirement: The symbolic side takes exact numbers only
A float SHALL be rejected wherever a symbolic value takes a number; exact values SHALL be written with integers and exact scalars.

#### Scenario: Float inside an expression
- **WHEN** `q` is a symbol and the code computes `q + 0.1`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

#### Scenario: Float angle
- **WHEN** the code builds the rotation about z by `math.pi`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

#### Scenario: Exact fraction
- **WHEN** the code computes `q + Scalar(1) / 10`
- **THEN** the result prints as `q + 1/10`

### Requirement: Simplification and LaTeX
A scalar SHALL provide simplification returning a scalar and a LaTeX rendering returning a string.

#### Scenario: Simplify a trigonometric identity
- **WHEN** the code simplifies `cos(q)**2 + sin(q)**2`
- **THEN** the result is the scalar `1`

#### Scenario: LaTeX output
- **WHEN** the code renders `cos(q)` as LaTeX
- **THEN** it obtains the string `\cos{\left(q \right)}`

### Requirement: Symbolic values have no equality
Comparing symbolic values with `==` or `!=`, hashing them and using them as a truth value SHALL raise an error saying to compile and compare the numbers at a configuration; this holds for scalars, points, vectors, rotations, transforms and axis-angle results.

#### Scenario: Equal-looking rotations
- **WHEN** the code compares two rotations about z by the same symbol `q`, built separately, with `==`
- **THEN** it raises an error naming the comparison and pointing to compiled values

#### Scenario: Scalar as a truth value or a key
- **WHEN** the code writes `if q:` or uses `q` as a dictionary key
- **THEN** it raises an error

### Requirement: No untyped algebra in public signatures
No public function, method or attribute of the library SHALL accept or return a sympy type, and code outside the library's boundary modules SHALL be rejected by the linter when it imports sympy.

#### Scenario: sympy import outside the boundary
- **WHEN** a module outside the boundary modules imports sympy
- **THEN** `uv run ruff check` reports `TID251` for that import

#### Scenario: Public signatures are sympy-free
- **WHEN** the public API is inspected by a test that walks every public signature and annotation
- **THEN** no annotation names a sympy type
