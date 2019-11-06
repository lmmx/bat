from sympy import Circle, Ellipse, Point
import numpy as np

USE_SYMPY_INTERSECTION = True
VISUALISE = True  # Display plot after running
SAVE_PLOT = False  # Will take priority over VISUALISE

# circle
cx, cy = (10, 20)
r = 15
# ellipse
ecx, ecy = (20, 20)
a = 20  # h_radius
b = 10  # v_radius

c = Circle(Point(cx, cy), r)
e = Ellipse(Point(ecx, ecy), a, b)

if USE_SYMPY_INTERSECTION:
    i = e.intersection(c)

c_eq = c.arbitrary_point()
e_eq = e.arbitrary_point()

if VISUALISE or SAVE_PLOT:
    import matplotlib.pyplot as plt
    from math import sin, cos, radians as rad

    fig = plt.figure()
    ax = plt.subplot()
    ax.set_aspect(1.0)
    plt.axis([0, 40, 40, 0])
    ax.set_title("Ellipse intersects a unit circle", size=22)
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)

    for t in np.arange(rad(0), rad(360), rad(2)):
        pcx, pcy = eval(str(c_eq.x)), eval(str(c_eq.y))
        pex, pey = eval(str(e_eq.x)), eval(str(e_eq.y))
        plt.scatter(pcx, pcy, s=10, color="red")
        plt.scatter(pex, pey, s=10, color="blue")

if USE_SYMPY_INTERSECTION:
    intersection_points = [(float(ii.x), float(ii.y)) for ii in i]
    intersection_t = [
        (np.arccos((ip[0] - ecx) / a), np.arcsin((ip[1] - ecy) / b))
        for ip in intersection_points
    ]
    for ip in intersection_points:
        plt.scatter(ip[0], ip[1], s=30, color="yellow")

# Solving the simple version algebraically with Sage
# p_4 = -a^2*(-Yc + b + yc)*(Yc + b - yc) + b^2*(-Xc + r + xc)^2
# p_3 = 4*a^2*r*(Yc - yc)
# p_2 = 2*a^2*(Yc^2 - 2*Yc*yc - b^2 + 2*r^2 + yc^2) - 2*b^2*(-Xc + r + xc)*(Xc + r - xc)
# p_1 = 4*a^2*r*(Yc - yc)
# p_0 = -a^2*(-Yc + b + yc)*(Yc + b - yc) + b^2*(Xc + r - xc)^2
p_4 = -a**2*(-cy + b + ecy)*(cy + b - ecy) + b**2*(-cx + r + ecx)**2
p_3 = 4*a**2*r*(cy - ecy)
p_2 = 2*a**2*(cy**2 - 2*cy*ecy - b**2 + 2*r**2 + ecy**2) - 2*b**2*(-cx + r + ecx)*(cx + r - ecx)
p_1 = 4*a**2*r*(cy - ecy)
p_0 = -a**2*(-cy + b + ecy)*(cy + b - ecy) + b**2*(cx + r - ecx)**2

# Evaluate the polynomial arguments with list comprehension:
p_coeff = [p_4, p_3, p_2, p_1, p_0]
t_roots = np.roots(p_coeff)

# Deduplicate the roots but without sorting
t_roots = [t_roots[ii].real for ii in np.sort(
    [[tt.real for tt in t_roots].index(t) for t in np.unique(np.real(t_roots))]
)]
# alist # Say we have a list of values but there are duplicates
# [2, 5, 3, 4, 3, 6]
# np.unique(alist)          # numpy.unique will sort after deduplicating,
# array([2, 3, 4, 5, 6])    # but the order may be important
# [alist[ii] for ii in np.sort([alist.index(i) for i in np.unique(alist)])]
# [2, 5, 3, 4, 6]
# list.index(i) will give the index of the first instance of a value in the list
# sort the list of these indices for every value in the deduplicated list to get
# just the first instance of each value (i.e. deduplicate without sorting)


if VISUALISE or SAVE_PLOT:
    pcol = ["yellow", "palegreen", "magenta", "chocolate"]

for t_num, t in enumerate(t_roots):
    pol = [
        (c[1] * t ** c[0]) for c in list(reversed(list(enumerate(reversed(p_coeff)))))
    ]
    if int(sum(pol)) != 0:
        print(f"Polynomial didn't evaluate to 0 as expected, root {t} not valid. Skipping...")
        continue # Do not plot this root, skip to the next root in t_roots (if any left)

    # Check with Sage
    # x = Xc + r*( (1 - t^2) / (1 + t^2) )
    ### When t = 0.7071067811865476:
    # 1.3333333333333335
    ### When t = -0.7071067811865476:
    # 1.3333333333333333
    x_crossing = cx + r * ((1 - t ** 2) / (1 + t ** 2))
    # y = Yc + r*( (2*t) / (1 + t^2) )
    ### When t = 0.7071067811865476:
    # 2.942809041582063
    ### When t = -0.7071067811865476:
    # 1.0571909584179364
    y_crossing = cy + r * ((2 * t) / (1 + t ** 2))

    if VISUALISE or SAVE_PLOT:
        plt.scatter(x_crossing, y_crossing, s=50, color=pcol[t_num])

if SAVE_PLOT:
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plt.savefig("simple-ellipse-sketch.png", bbox_inches="tight")
elif VISUALISE:
    plt.show()
