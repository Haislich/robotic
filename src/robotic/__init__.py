import sympy

Scalar = int | float | sympy.Expr | sympy.Function | sympy.Basic


def scalar_repr_latex(name: str, value: Scalar):
    if hasattr(value, "evalf"):
        value = value.evalf()  # type: ignore
    if hasattr(value, "is_symbolic"):
        if not value.is_symbolic():  # type: ignore
            value = round(value, 4)  # type: ignore

    if hasattr(value, "_repr_latex_"):
        return rf"${name} = {value._repr_latex_().replace(r'$', '')}$"  # type: ignore
    else:
        return rf"${name} = {value}$"
