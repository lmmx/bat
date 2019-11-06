from sympy import Circle, Ellipse, Point
import numpy as np

USE_SYMPY_INTERSECTION = True
# Set VISUALISE to True to view plot, it's False
# so that outputs of this script can be imported
# without overwriting the plot being built in sketch.py
VISUALISE = True

# circle
cx, cy = (550, 1000)
r = 225
# ellipse
ecx, ecy = (1000, 1000)
a = 675 # h_radius
b = 275 # v_radius
# N.B.: When comparing to variables in sketch.py that
# h_radius/v_radius there refer to its outer ellipse

c = Circle(Point(cx, cy), r)
e = Ellipse(Point(ecx, ecy), a, b)

if USE_SYMPY_INTERSECTION:
    i = e.intersection(c)

    intersection_points = [(float(ii.x), float(ii.y)) for ii in i]
    intersection_t = [
        (np.arccos((ip[0] - ecx) / a), np.arcsin((ip[1] - ecy) / b))
        for ip in intersection_points
    ]

if VISUALISE:
    from math import sin, cos, radians as rad
    import matplotlib.pyplot as plt

    ax = plt.subplot()
    ax.set_aspect(1.0)
    plt.axis([0, 2000, 2000, 0])

    c_eq = c.arbitrary_point()
    e_eq = e.arbitrary_point()

    # Draw the circle in red and the ellipse in blue
    for t in np.arange(rad(0), rad(360), rad(2)):
        pcx, pcy = eval(str(c_eq.x)), eval(str(c_eq.y))
        pex, pey = eval(str(e_eq.x)), eval(str(e_eq.y))
        plt.scatter(pcx, pcy, s=10, color="red")
        plt.scatter(pex, pey, s=10, color="blue")

# Solving the intersection algebraically with Sage
# p_4 = a^2 * ((Yc - yc)^2 - b^2) + b^2 * ( (Xc - xc) - r )^2
# 0
p_4 = a ** 2 * ((cy - ecy) ** 2 - b ** 2) + b ** 2 * ((cx - ecx) - r) ** 2
# p_3 = 4 * a^2 * r * ( Yc - yc )
# 0
p_3 = 4 * a ** 2 * r * (cy - ecy)
# p_2 = 2 * ( a^2 * ( (Yc - yc)^2 - b^2 + 2*r^2 ) ) + b^2 * ( (Xc - xc) + r )^2
# 27179296875
p_2 = (
    2 * (a ** 2 * ((cy - ecy) ** 2 - b ** 2 + 2 * r ** 2))
    + b ** 2 * ((cx - ecx) + r) ** 2
)
# p_1 = 4 * a^2 * r * (Yc - yc)
# 0
p_1 = 4 * a ** 2 * r * (cy - ecy)
# p_0 = a^2 * ((Yc - yc)^2 - b^2) + b^2 * ((Xc - xc) + r)^2
# -30628125000
p_0 = a ** 2 * ((cy - ecy) ** 2 - b ** 2) + b ** 2 * ((cx - ecx) + r) ** 2
#
# Evaluate the polynomial arguments with list comprehension:
p_coeff = [p_4, p_3, p_2, p_1, p_0]
t_roots = np.roots(p_coeff)

if VISUALISE:
    pcol = ["yellow", "palegreen", "magenta", "chocolate"]

for t_num, t in enumerate(t_roots):
    pol = [
        (c[1] * t ** c[0]) for c in list(reversed(list(enumerate(reversed(p_coeff)))))
    ]
    assert (
        int(sum(pol)) == 0
    ), "Polynomial didn't evaluate to 0 as expected, root not found"

    # Check with Sage
    # x = Xc + r*( (1 - t^2) / (1 + t^2) )
    ### When t = 1.0615515694374378:
    # 536.576354560990
    ### When t = -1.0615515694374378:
    # 536.576354560990
    x_crossing = cx + r * ((1 - t ** 2) / (1 + t ** 2))
    # y = Yc + r*( (2*t) / (1 + t^2) )
    ### When t = 0.7071067811865476:
    # 1224.59921135910
    ### When t = -1.0615515694374378:
    # 775.400788640904
    y_crossing = cy + r * ((2 * t) / (1 + t ** 2))

    if VISUALISE:
        plt.scatter(x_crossing, y_crossing, s=50, color=pcol[t_num])

if VISUALISE:
    if USE_SYMPY_INTERSECTION:
        pcol = ["gray", "lime", "magenta"]
        for ip in enumerate(intersection_t):
            alpha_x = ip[1][0] # use the arccos derived value
            alpha_y = ip[1][1] # use the arcsin derived value
            el_x = ecx + a * cos(alpha_x)
            el_y = ecy + b * sin(alpha_y)
            plt.scatter(el_x, el_y, s=50, color=pcol[ip[0]])
