# Spec Delta

## Purpose

Provides the typed constants and expressions every other layer is built on, whose type is known to the type checker, so exam-style formulas stay typed from the first symbol to the last simplification, and so the type says whether a value is fixed, the only case where zero and equality have meaning.

## ADDED Requirements

### Requirement: Typed scalar arithmetic
Constants and expressions SHALL support addition, subtraction, multiplication, division, power and negation with a constant, an expression or an `int`, on either side; the result SHALL be typed as a constant when every operand is a constant or an `int`, and as an expression otherwise, whatever the result simplifies to.

#### Scenario: Mixed arithmetic stays typed
- **WHEN** `q` is a symbol and the code computes `q + 1`, `1 + q`, `q * q`, `2 * q - Scalar(1) / 2` and `-q`
- **THEN** the type checker infers an expression for each result

#### Scenario: Constants stay constants
- **WHEN** the code computes `Scalar(2) + 3`, `Scalar.pi() / 2` and `Scalar(2).sqrt()`
- **THEN** the type checker infers a constant for each result

#### Scenario: A cancelled symbol stays an expression
- **WHEN** `q` is a symbol and the code computes `q - q`
- **THEN** the result prints as `0` and is an expression, for the type checker and at runtime

### Requirement: Constant or symbolic is in the type
Every point, vector, rotation, transform and axis-angle result SHALL carry in its type whether it is constant or symbolic; it SHALL be constant only when built from constants only, and its components SHALL then be constants; otherwise every component SHALL be an expression, constant ones included.

#### Scenario: Positions
- **WHEN** `a` is the constant 2 and `b` a symbol, and the code builds the positions (a, a, a), (a, a, b) and (a, a, a + b) in `B1`
- **THEN** the type checker infers a constant position for the first and a symbolic position for the other two

#### Scenario: A constant component of a symbolic position
- **WHEN** the code reads the x component of the position (a, a, b)
- **THEN** it is an expression printing `2`, for the type checker and at runtime

#### Scenario: Rotation and transform
- **WHEN** the code builds the rotation about z by π/2 and the rotation about z by a symbol `q`, and the homogeneous matrix of the first with the translation (0, 0, b)
- **THEN** the type checker infers a constant rotation, a symbolic rotation and a symbolic transform

### Requirement: Constants compare with a tolerance
A constant SHALL compare with another constant or an `int` through a method returning whether their difference, evaluated to 30 significant digits, is within an absolute tolerance, `1e-9` unless the caller passes another; expressions SHALL have no such method.

#### Scenario: A zero no simplification proves
- **WHEN** the code compares cos(π/7) + cos(3π/7) + cos(5π/7) with the constant 1/2
- **THEN** the answer is true

#### Scenario: Chosen tolerance
- **WHEN** the code compares the constants 1 and 1 + 1/1000 with the tolerance `1e-2`, and then with the default tolerance
- **THEN** the first answer is true and the second false

#### Scenario: An expression has no tolerance comparison
- **WHEN** `q` is a symbol and the code compares `q` with 0 by the tolerance method
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Evaluation substitutes exact values
An expression, and every symbolic point, vector, rotation, transform and axis-angle result, SHALL provide evaluation taking exact values by symbol name, possibly none, and returning the constant counterpart; a symbol without a value SHALL raise naming it, values for symbols that do not occur SHALL be allowed, and a float value SHALL raise.

#### Scenario: Nothing to substitute
- **WHEN** the code evaluates the x component of the position (a, a, b) with no values
- **THEN** the result is the constant 2

#### Scenario: Substituting a symbol
- **WHEN** the code evaluates the z component of that position with `b` set to 1
- **THEN** the result is the constant 1

#### Scenario: A whole position
- **WHEN** the code evaluates the position (a, a, b) with `b` set to 1 and an unused symbol set to 5
- **THEN** the type checker infers a constant position, and it is (2, 2, 1)

#### Scenario: Missing value
- **WHEN** the code evaluates the z component of that position with no values
- **THEN** an error naming `b` is raised

#### Scenario: Float value
- **WHEN** the code evaluates it with `b` set to `0.5`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

#### Scenario: Non-numbers are rejected
- **WHEN** the code computes `q + "a"`
- **THEN** the type checker reports an error
- **AND** executing it raises an error naming the mistake

### Requirement: Symbols are created only by the library
The library SHALL provide the only way to create a symbol, from a name, with fixed assumptions, so that two symbols with the same name are the same symbol.

#### Scenario: Same name, same symbol
- **WHEN** `q2` is created twice by name and the two are added
- **THEN** the result is `2*q2` with one free symbol

#### Scenario: A variable is required
- **WHEN** a function requires a variable and the code passes it `q + 1`
- **THEN** the type checker reports an error

### Requirement: Exact constants
The library SHALL provide exact π, square root and exponential as scalar operations, so exact values never pass through a float.

#### Scenario: Exact pi
- **WHEN** the code computes π divided by 2 through the library
- **THEN** the result prints as `pi/2` and contains no float

#### Scenario: Exact square root
- **WHEN** the code takes the square root of the scalar 2
- **THEN** the result prints as `sqrt(2)`

### Requirement: Elementary functions as methods
Constants and expressions SHALL provide cosine, sine, tangent, two-argument arctangent, square root and exponential as methods, returning a constant on a constant and an expression on an expression.

#### Scenario: Trigonometry
- **WHEN** `q` is a symbol and the code calls its cosine method
- **THEN** the result is an expression printing as `cos(q)`

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
Constants and expressions SHALL provide simplification returning a value of the same class and a LaTeX rendering returning a string. Simplification SHALL happen only when the code calls it: no operation, construction, extraction, compilation or LaTeX rendering SHALL simplify.

#### Scenario: Simplify a trigonometric identity
- **WHEN** the code simplifies `cos(q)**2 + sin(q)**2`
- **THEN** the result is an expression printing as `1`

#### Scenario: LaTeX output
- **WHEN** the code renders `cos(q)` as LaTeX
- **THEN** it obtains the string `\cos{\left(q \right)}`

#### Scenario: LaTeX does not simplify
- **WHEN** the code renders `cos(q)**2 + sin(q)**2` as LaTeX
- **THEN** the string shows both terms, not `1`

### Requirement: Symbolic values have no equality
Comparing symbolic values with `==` or `!=`, hashing them and using them as a truth value SHALL raise an error saying what to do instead: the tolerance comparison for constants, evaluation or compilation for expressions; this holds for constants, expressions, points, vectors, rotations, transforms and axis-angle results.

#### Scenario: Equal constants
- **WHEN** the code compares `Scalar(2)` with `Scalar(2)` using `==`
- **THEN** it raises an error pointing to the tolerance comparison

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
