"""
PySR's complexity metric, evaluated on a sympy expression.
"""
from __future__ import annotations

import sympy as sp

# The search alphabet: what PySR was actually given in run_pysr_fold.
BINARY_OPERATORS = ("+", "-", "*", "/")
UNARY_OPERATORS = ("log", "exp", "sqrt", "asin", "sinh")

_FUNC_NAME = {
    sp.log: "log", sp.exp: "exp", sp.asin: "asin", sp.sinh: "sinh",
    sp.tan: "tan", sp.cos: "cos", sp.sin: "sin", sp.cosh: "cosh",
    sp.tanh: "tanh", sp.acos: "acos", sp.atan: "atan", sp.Abs: "abs",
}


def _is_neg_one(a: sp.Expr) -> bool:
    return a.is_Number and a == -1


def complexity(expr) -> int:
    """Node count of `expr` in PySR's convention."""
    e = sp.sympify(expr)

    if e.is_Atom:
        return 1                                    # symbol, number, or pi

    if e.is_Add:
        # n-1 binary nodes, and a term written -x is a subtraction: its -1 is the
        # operator, not a constant of its own.
        n = len(e.args) - 1
        for a in e.args:
            if a.is_Mul and _is_neg_one(a.args[0]):
                a = sp.Mul(*a.args[1:], evaluate=False)
            n += complexity(a)
        return n

    if e.is_Pow:
        b, p = e.args
        if p == 1:
            return complexity(b)                    # an unevaluated x**1 is just x
        if p == sp.Rational(1, 2):
            return 1 + complexity(b)                # sqrt: one unary node
        if p.is_Integer and p > 1:
            return (int(p) - 1) + int(p) * complexity(b)      # u**3 -> u*u*u
        if p.is_negative:
            return 1 + 1 + complexity(sp.Pow(b, -p))          # 1/x -> `/`(1, x)
        return 1 + complexity(b) + complexity(p)    # outside the alphabet; see below

    if e.is_Mul:
        args = list(e.args)
        if _is_neg_one(args[0]):
            return 1 + complexity(sp.Mul(*args[1:], evaluate=False))
        # Denominators are found structurally, as the factors carrying a negative power.
        # as_numer_denom() would be the obvious way to do this and is wrong here: it
        # re-evaluates, distributing a constant factor over a sum, so `(a + b)*c` would
        # be counted as the 5-node `a*c + b*c` rather than the 3-node product PySR built.
        num, den = [], []
        for a in args:
            if a.is_Pow and a.args[1].is_number and a.args[1].is_negative:
                b_, p_ = a.args
                den.append(b_ if p_ == -1 else sp.Pow(b_, -p_, evaluate=False))
            else:
                num.append(a)
        if den:
            n = sp.Mul(*num, evaluate=False) if num else sp.Integer(1)
            d = sp.Mul(*den, evaluate=False) if len(den) > 1 else den[0]
            return 1 + complexity(n) + complexity(d)          # one `/` node
        return (len(args) - 1) + sum(complexity(a) for a in args)

    if isinstance(e, sp.Function):
        return 1 + sum(complexity(a) for a in e.args)

    raise TypeError(f"cannot count node {e.func.__name__}: {e}")


def leaf_multiplicity(expr, sym: sp.Symbol) -> int:
    """How many leaves `sym` occupies in the tree `complexity` counts.

    Not simply the number of times `sym` occurs. There is no `^` among the binary
    operators, so an integer power is counted as repeated multiplication and u**3
    contributes three leaves, not one. Exponents therefore multiply along the path from
    the root, which is what makes the substitution law in sr_simplify.compose_raw exact
    rather than approximate.
    """
    e = sp.sympify(expr)

    if e.is_Atom:
        return 1 if e == sym else 0

    if e.is_Pow:
        b, p = e.args
        if p == 1:
            return leaf_multiplicity(b, sym)
        if p == sp.Rational(1, 2):
            return leaf_multiplicity(b, sym)
        if p.is_Integer and p > 1:
            return int(p) * leaf_multiplicity(b, sym)
        if p.is_negative:
            return leaf_multiplicity(sp.Pow(b, -p, evaluate=False), sym)
        return leaf_multiplicity(b, sym) + leaf_multiplicity(p, sym)

    return sum(leaf_multiplicity(a, sym) for a in e.args)


def operators_used(expr) -> set[str]:
    """Every operator in `expr`, named as PySR names them.

    Anything here that is not in BINARY_OPERATORS or UNARY_OPERATORS could not have been
    produced by the search: it entered through the change of variable (tan and pi do,
    via tan_pi) or through a simplification that introduced a non-integer power.
    """
    e = sp.sympify(expr)
    out: set[str] = set()

    def walk(x: sp.Expr) -> None:
        if x.is_Atom:
            return
        if x.is_Add:
            out.add("-" if any(a.is_Mul and _is_neg_one(a.args[0]) for a in x.args)
                    else "+")
        elif x.is_Pow:
            _, p = x.args
            if p == sp.Rational(1, 2):
                out.add("sqrt")
            elif p.is_Integer and p > 1:
                out.add("*")
            elif p.is_negative:
                out.add("/")
            else:
                out.add("^")
        elif x.is_Mul:
            _, den = x.as_numer_denom()
            out.add("/" if den != sp.S.One else "*")
        elif isinstance(x, sp.Function):
            out.add(_FUNC_NAME.get(type(x), type(x).__name__))
        for a in x.args:
            walk(a)

    walk(e)
    return out


def outside_alphabet(expr) -> set[str]:
    """Operators in `expr` that the search could not have generated."""
    allowed = set(BINARY_OPERATORS) | set(UNARY_OPERATORS)
    return operators_used(expr) - allowed
