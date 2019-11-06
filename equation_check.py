from sympy import var
from collections import OrderedDict

# Register all symbols used in the parametric simultaneous equation
a, b, r, t, Xc, Yc, xc, yc = var("a, b, r, t, Xc, Yc, xc, yc")

a_sq_expr = a ** 2 * ((1 + t ** 2) * (Yc - yc) + 2 * r * t) ** 2
b_sq_expr = b ** 2 * (r * (1 - t ** 2) + (Xc - xc) * (t ** 2 + 1)) ** 2
rhs_expr = a ** 2 * b ** 2 * (1 + t ** 2) ** 2

expr = sum([a_sq_expr, b_sq_expr, -rhs_expr])


def pretty_expr(expr, expand=False, newlines=False, unicode_powers=True):
    if expand:
        expr = expr.expand()
    s = str(expr).replace("**", "^").replace("*", "·")
    if unicode_powers:
        s = s.replace("^2", "²")
    if newlines:
        s = s.replace(" +", "\n+").replace(" -", "\n-")
    return s


def collect_by_var(expr, var):
    var_poly_terms = [term.as_poly(var) for term in expr.expand().args]
    vars_by_coeff = dict()
    for p in var_poly_terms:
        for (n, coeff) in enumerate(reversed(p.all_coeffs())):
            if coeff != 0:
                if n not in vars_by_coeff.keys():
                    vars_by_coeff[n] = []
                vars_by_coeff[n].append(coeff)
    return OrderedDict(sorted(vars_by_coeff.items()))


def factor_expr(expr_arg_list, var=None, group_ab_by_a=True, custom_ab=var("a, b")):
    # The negative `not has()` for b_terms avoids counting a*b terms twice
    a, b = custom_ab  # For some reason this was the only way to set (a, b) changeably
    if group_ab_by_a:
        a_terms = [a_term for a_term in expr_arg_list if a_term.has(a)]
        b_terms = [
            b_term for b_term in expr_arg_list if b_term.has(b) and not b_term.has(a)
        ]
    else:
        a_terms = [
            a_term for a_term in expr_arg_list if a_term.has(a) and not a_term.has(b)
        ]
        b_terms = [b_term for b_term in expr_arg_list if b_term.has(b)]
    assert len(a_terms + b_terms) == len(expr_arg_list), (
        f"WARNING: Terms were missed!"
        + f" Expected to factor {len(expr_arg_list)} terms,"
        + f" but only got {len(a_terms + b_terms)}."
        + "\nYou can set up to two factors to group by with the custom_ab=(a,b), "
        + "if there's more than 2 then please change the code."
        + "This code was designed for ellipse equations, which have only a and b factors"
    )
    factored = 0  # The empty expression
    # Skip if empty list - avoids error from `sum([]).factor()` becoming `0.factor()`
    if a_terms:
        factored += sum(a_terms).factor(var)
    if b_terms:
        factored += sum(b_terms).factor(var)
    return factored


def test_factor_expr(expr=expr, var=t):
    collected = collect_by_var(expr, var)
    for k in collected.keys():
        assert factor_expr(collected[k]).expand() - sum(collected[k]).expand() == 0
    return


def print_collected_expr(expr=expr, var=t, group_ab_by_a=True):
    collected = collect_by_var(expr, var)
    for k in collected.keys():
        power_expr = sum(collected[k]).simplify()
        print(f"{var}^{k}:")
        print(f"Original        {pretty_expr(sum(collected[k]))}")
        # print(f"Simplified      {pretty_expr(sum(collected[k]).simplify())}")
        print(
            f"Collected       {pretty_expr(sum(collected[k]).collect([a,b]).simplify())}"
        )
        print(
            f"Factored        {pretty_expr(factor_expr(collected[k], group_ab_by_a=group_ab_by_a))}\n"
        )
    return


def print_p_coeff(
    expr=expr,
    var=var,
    replace_var_names=False,
    replace_dict=None,
    sage_format=False,
    pretty=False,
    group_ab_by_a=True,
):
    if replace_var_names and replace_dict is None:
        replace_dict = OrderedDict({"Xc": "cx", "xc": "ecx", "Yc": "cy", "yc": "ecy"})
    collected = collect_by_var(expr, t)
    for k in reversed(collected.keys()):
        p_coeff_str = str(factor_expr(collected[k], group_ab_by_a=group_ab_by_a))
        if replace_var_names:
            for target_var in replace_dict.keys():
                p_coeff_str = p_coeff_str.replace(target_var, replace_dict[target_var])
        if pretty:
            p_coeff_str = pretty_expr(p_coeff_str)
        elif sage_format:
            p_coeff_str = p_coeff_str.replace("**", "^")
        print(f"p_{k} = {p_coeff_str}")
    return
