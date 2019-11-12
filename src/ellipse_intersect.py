from sympy import Circle, Ellipse, Point
import numpy as np

# Set VISUALISE to True to view plot, it's False
# so that outputs of this script can be imported
# without overwriting the plot being built in sketch.py
VISUALISE = False
EXPORT_INTERSECTIONS = True
# USE_SYMPY_INTERSECTION = False # Deprecated

if EXPORT_INTERSECTIONS:
    intersection_points = []
    intersection_t = []

# circle
cx, cy = (550, 1000)
r = 225
# ellipse
ecx, ecy = (1000, 1000)
a = 675  # h_radius
b = 275  # v_radius
# N.B.: When comparing to variables in sketch.py that
# h_radius/v_radius there refer to its outer ellipse

c = Circle(Point(cx, cy), r)
e = Ellipse(Point(ecx, ecy), a, b)

# if USE_SYMPY_INTERSECTION:
#     i = e.intersection(c)
#
#     intersection_points = [(float(ii.x), float(ii.y)) for ii in i]
#     intersection_t = [
#         (np.arccos((ip[0] - ecx) / a), np.arcsin((ip[1] - ecy) / b))
#         for ip in intersection_points
#     ]

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

# Solving the simple version algebraically with Sage
# p_4 = -a^2*(-Yc + b + yc)*(Yc + b - yc) + b^2*(-Xc + r + xc)^2
# p_3 = 4*a^2*r*(Yc - yc)
# p_2 = 2*a^2*(Yc^2 - 2*Yc*yc - b^2 + 2*r^2 + yc^2) - 2*b^2*(-Xc + r + xc)*(Xc + r - xc)
# p_1 = 4*a^2*r*(Yc - yc)
# p_0 = -a^2*(-Yc + b + yc)*(Yc + b - yc) + b^2*(Xc + r - xc)^2
p_4 = -a ** 2 * (-cy + b + ecy) * (cy + b - ecy) + b ** 2 * (-cx + r + ecx) ** 2
p_3 = 4 * a ** 2 * r * (cy - ecy)
p_2 = 2 * a ** 2 * (
    cy ** 2 - 2 * cy * ecy - b ** 2 + 2 * r ** 2 + ecy ** 2
) - 2 * b ** 2 * (-cx + r + ecx) * (cx + r - ecx)
p_1 = 4 * a ** 2 * r * (cy - ecy)
p_0 = -a ** 2 * (-cy + b + ecy) * (cy + b - ecy) + b ** 2 * (cx + r - ecx) ** 2

# Evaluate the polynomial arguments with list comprehension:
p_coeff = [p_4, p_3, p_2, p_1, p_0]
t_roots = np.roots(p_coeff)

# Deduplicate the roots but without sorting
t_roots = [
    t_roots[ii].real
    for ii in np.sort(
        [[tt.real for tt in t_roots].index(t) for t in np.unique(np.real(t_roots))]
    )
]
# alist # Say we have a list of values but there are duplicates
# [2, 5, 3, 4, 3, 6]
# np.unique(alist)          # numpy.unique will sort after deduplicating,
# array([2, 3, 4, 5, 6])    # but the order may be important
# [alist[ii] for ii in np.sort([alist.index(i) for i in np.unique(alist)])]
# [2, 5, 3, 4, 6]
# list.index(i) will give the index of the first instance of a value in the list
# sort the list of these indices for every value in the deduplicated list to get
# just the first instance of each value (i.e. deduplicate without sorting)


if VISUALISE:
    pcol = ["yellow", "palegreen", "magenta", "chocolate"]

for t_num, t in enumerate(t_roots):
    pol = [
        (c[1] * t ** c[0]) for c in list(reversed(list(enumerate(reversed(p_coeff)))))
    ]
    if int(sum(pol)) != 0:
        print(
            f"Polynomial didn't evaluate to 0 as expected. "
            + f"Root {t} not valid. Skipping..."
        )
        continue  # Do not plot this root, skip to next root in t_roots (if any left)

    x_crossing = cx + r * ((1 - t ** 2) / (1 + t ** 2))
    y_crossing = cy + r * ((2 * t) / (1 + t ** 2))

    if VISUALISE:
        plt.scatter(x_crossing, y_crossing, s=50, color=pcol[t_num])
    if EXPORT_INTERSECTIONS:
        intersection_points.append((x_crossing, y_crossing))
        intersection_t.append(t)

if VISUALISE:
    # if USE_SYMPY_INTERSECTION:
    #    pcol = ["gray", "lime", "magenta"]
    #    for ip in enumerate(intersection_t):
    #        alpha_x = ip[1][0]  # use the arccos derived value
    #        alpha_y = ip[1][1]  # use the arcsin derived value
    #        el_x = ecx + a * cos(alpha_x)
    #        el_y = ecy + b * sin(alpha_y)
    #        plt.scatter(el_x, el_y, s=50, color=pcol[ip[0]])
    plt.show()
