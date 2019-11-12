import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, arccos, array, array_equal, arctan2, dot, subtract
from numpy.linalg import norm
from math import sin, cos, radians as rad, sqrt
from ellipse_intersect import intersection_points, intersection_t

VISUALISE = True
SAVE_PLOT = False

if VISUALISE or SAVE_PLOT:
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_aspect(1.0)
    plt.axis([6, 18, 14, 2])
    ax.set_title("Interior path enclosed by some circles", size=18)
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)

scale_factor = 1

# On sketch diagram these are a:b ≈ 9:5
h_radius, v_radius = np.dot(scale_factor, [9, 5])
focal_len = sqrt(h_radius ** 2 - v_radius ** 2)

xc = 10
yc = 10

if VISUALISE or SAVE_PLOT:
    plt.scatter(xc, yc, s=10, color="k")

    arrow_opacity = 0.4

    # Draw the outer ellipse using measurements obtained by eye
    for alpha in np.arange(rad(0), rad(360), rad(3)):
        oe_x = xc + h_radius * cos(alpha)
        oe_y = yc + v_radius * sin(alpha)
        plt.scatter(oe_x, oe_y, s=2, color="k")

# Determine top circle radii
# For now, just take it as 90% of v_radius
tc_scale = 0.9
tc_r = 0.5 * v_radius * tc_scale

lower_circle_scale = 0.6  # Relative to the circles on the top half of the ellipse
lc_r = tc_r * lower_circle_scale  # Radius of the lower circle

# Store the circle centres, these will be used together later in the calculation
# of centre point adjacencies (i.e. which circles neighbour one another)

# circle_centres = [(14, 12), (13, 14), (12, 16), (10,16), (8, 16), (7, 14), (6, 12), (5, 9), (5, 6), (8.5, 5), (11.5, 5), (15, 6), (15, 9.5)]
circle_centres = [(14, 12), (12, 12), (10, 10), (8.5, 8), (11, 5), (15, 6), (15, 9.5)]

# Plot the circles

if VISUALISE or SAVE_PLOT:
    # CIRCLE i
    for i in np.arange(0, len(circle_centres)):
        cxc, cyc = circle_centres[i]
        if cyc > yc:
            c_r = lc_r
        else:
            c_r = tc_r
        for alpha in np.arange(rad(0), rad(360), rad(1)):
            c_x = cxc + c_r * cos(alpha)
            c_y = cyc + c_r * sin(alpha)
            plt.scatter(c_x, c_y, s=6, color="pink")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# All circles have now been drawn and their interior arcs delineate the
# outline of the bat. Success! Next, to trace out this outline, first
# construct an adjacency matrix of the centre points

circle_centres_clockwise = circle_centres

# An alternative way to do this is taking all-against-all distance transform of the
# matrix of circle centres, i.e. finding a permutation to relabel the circle centres
# according to proximity. This is useful even if it's possible to do by hand.


def dist(p, q):
    """
    Calculate the Euclidean distance between two points (p,q)
    """
    return norm(subtract(p, q))


def choose_interior_ip(c1c, c2c, interior_centre=(xc, yc), tc_r=tc_r, lc_r=lc_r):
    """
    Provide the centre coordinates for two circles, from which the radius will be
    derived according to the default radius above/below the ellipse midpoint.
    """
    assert len(c1c) == len(c2c) == 2
    d = dist(c1c, c2c)
    c1x, c1y = c1c
    c2x, c2y = c2c
    c1r, c2r = tc_r, tc_r
    # Reassign radius as top circle radius if circle centre is below cy midpoint
    if c1y > yc:
        c1r = lc_r
    if c2y > yc:
        c2r = lc_r
    assert not d > c1r + c2r, (
        f"The circles at ({c1x},{c1y}) and ({c2x},{c2y}) do not intersect!"
        + f" The radii sum to {c1r+c2r} but the circles are {d} apart"
    )

    # Let circle 1 be imagined to be at (0,0) of a new basis and circle 2 at (d, 0)
    # We will translate and rotate back to the coordinate system after calculating a,
    # the distance to the intersection chord at (a, 0), and b, the half-length of the
    # chord between the two intersection points/cusps of the lens at (a,b) and (a,-b)
    # (d - a)^2 + (c1r^2 - a^2) = c2r^2
    # ==> d^2 - 2*d*a + a^2 + c1r^2 - a^2 = c2r^2
    # ==> a = (d^2 + c1r^2 - c2r^2) / (2*d)
    # Note that a here is not an absolute coordinate, just a relative distance
    # a = (d**2 + c1r**2 - c2r**2) / (2*d)
    a = (d ** 2 + c1r ** 2 - c2r ** 2) / (
        2 * d
    )  # distance from c1c to intersecting chord
    b = sqrt(c1r ** 2 - a ** 2)  # half-length of the intersecting chord

    # Construct a basis
    AB = subtract(c2c, c1c)  # AB = B - A
    e1 = AB / d  # e1 = AB / |AB|; norm(e1) = 1
    assert (
        round(norm(e1), ndigits=5) == 1
    ), "Error: unit vector e1 = AB/|AB| should be 1"
    e2 = (
        array([[0, -1], [1, 0]]) @ e1
    )  # 90° rotation so e2 perp. e1 (orthonormal basis)

    possible_ip = []

    # P_{1,2} = c1c + a·e1 +/- b·e2
    for b_component in [b, -b]:
        ip = (c1x, c1y) + a * e1 + b_component * e2
        test_c1 = round((ip[0] - c1x) ** 2 + (ip[1] - c1y) ** 2, ndigits=5)
        test1 = test_c1 == round(c1r ** 2, ndigits=5)
        assert test1, f"Error: circle {cc_n} intersection {ip} is not on the circle"
        +f" at {(c1x,c1y)}: {test_c1}≠{c1r**2}"
        test_c2 = round((ip[0] - c2x) ** 2 + (ip[1] - c2y) ** 2, ndigits=5)
        test2 = test_c2 == round(c2r ** 2, ndigits=5)
        assert test2, f"Error: circle {cc_n} intersection {ip} is not on the circle"
        +f" at {(c2x,c2y)}: {test_c2}≠{c2r**2}"
        possible_ip.append(ip)

    # Then compare the [at most] two points and select the nearest to interior_centre
    ip_dists = [dist(interior_centre, ip) for ip in possible_ip]
    ip_n = np.array(ip_dists).argsort().argsort().tolist().index(0)
    ip = possible_ip[ip_n]
    c1t = arctan2(ip[1] - c1y, ip[0] - c1x)
    if c1t < 0:
        assert (c1t + 2 * np.pi) > 0, f"arctan2 returned a number below -2*pi ({c1t})"
        c1t += 2 * np.pi
    rec_x = c1x + (c1r * cos(c1t))
    rec_y = c1y + (c1r * sin(c1t))
    rec = (rec_x, rec_y)
    assert array_equal(
        np.round(rec, 5), np.round(ip, 5)
    ), f"Failed to recover intersection {ip} (got {(rec)})"
    print(f"Recovered {ip} as {rec}: t = {c1t} (circle at {c1c}, r={c1r})")
    return ip, c1t


color_list = ["red", "blue", "k", "lime"]
for cc_n, (cc_xc, cc_yc) in enumerate(circle_centres_clockwise):
    # Assign radius based on whether in upper or lower half of the ellipse
    if cc_yc > yc:
        cc_r = lc_r
    else:
        cc_r = tc_r
    prev_xc, prev_yc = circle_centres_clockwise[cc_n - 1]
    next_xc, next_yc = circle_centres_clockwise[(cc_n + 1) % len(circle_centres)]
    # Each adjacent circle (2 per circle) may have 1 or 2 intersection points
    ic = (12.3, 8)  # This is the interior centre for this case study example
    prev_ip, prev_t = choose_interior_ip((cc_xc, cc_yc), (prev_xc, prev_yc), ic)
    next_ip, next_t = choose_interior_ip((cc_xc, cc_yc), (next_xc, next_yc), ic)
    if VISUALISE or SAVE_PLOT:
        start_t, end_t = sorted((prev_t, next_t))
        # Always plot the minimal length arc (corresponds to ellipse interior)
        if (prev_t + rad(360) - next_t) < (next_t - prev_t):
            # In this case, invert the expected order
            start_t += rad(360)
        start_t, end_t = sorted((start_t, end_t))
        print(
            f"Drawing arc on circle {cc_n} (centre {(cc_xc, cc_yc)} from {start_t} to {end_t}"
        )
        for arc_t in np.arange(start_t, end_t, rad(2)):
            arc_x = cc_xc + cc_r * cos(arc_t)
            arc_y = cc_yc + cc_r * sin(arc_t)
            plt.scatter(
                arc_x, arc_y, s=50, color=color_list[cc_n % len(color_list)], alpha=0.3
            )
        plt.scatter(prev_ip[0], prev_ip[1], s=30, color="orange")
        plt.scatter(next_ip[0], next_ip[1], s=30, color="yellow")


if SAVE_PLOT:
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plt.savefig("interior_trace_sketch.png", bbox_inches="tight")
elif VISUALISE:
    plt.show()
