import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, arccos, array, array_equal, dot, subtract
from numpy.linalg import norm
from math import sin, cos, radians as rad, sqrt

VISUALISE = True
SAVE_PLOT = False

if VISUALISE or SAVE_PLOT:
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_aspect(1.0)
    plt.axis([-10, 10, 6, -6])
    ax.set_title("Sketch of some intersecting circles", size=18)
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)

scale_factor = 1

# On sketch diagram these are a:b ≈ 9:5
h_radius, v_radius = dot(scale_factor, [9, 5])
focal_len = sqrt(h_radius ** 2 - v_radius ** 2)

x_offset, y_offset = 10, 10

xc = 10 - x_offset
yc = 10 - y_offset

if VISUALISE or SAVE_PLOT:
    plt.scatter(xc, yc, s=10, color="k")

    arrow_opacity = 0.4

    # Draw the outer ellipse using measurements obtained by eye
    for alpha in arange(rad(0), rad(360), rad(3)):
        oe_x = xc + h_radius * cos(alpha)
        oe_y = yc + v_radius * sin(alpha)
        plt.scatter(oe_x, oe_y, s=2, color="k")

# Determine top circle radii
# For now, just take it as 90% of v_radius
tc_scale = 0.4
tc_r = 1 * v_radius * tc_scale

lower_circle_scale = 0.6  # Relative to the circles on the top half of the ellipse
lc_r = tc_r * lower_circle_scale  # Radius of the lower circle

# Store the circle centres, these will be used together later in the calculation
# of centre point adjacencies (i.e. which circles neighbour one another)

circle_centres = [(0, 0), (3, -1)]

# Plot the circles

if VISUALISE or SAVE_PLOT:
    # CIRCLE i
    for i in arange(0, len(circle_centres)):
        cxc, cyc = circle_centres[i]
        if cyc > yc:
            c_r = lc_r
        else:
            c_r = tc_r
        for alpha in arange(rad(0), rad(360), rad(1)):
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
    return np.linalg.norm(subtract(p, q))


c1x, c1y = circle_centres[0]
c2x, c2y = circle_centres[1]
c1r, c2r = tc_r, tc_r
d = dist((c1x, c1y), (c2x, c2y))
interior_centre = (xc, yc)
# Reassign radius as smaller radius if circle centre is below cy midpoint
if c1y > interior_centre[1]:
    c1r = lc_r
if c2y > interior_centre[1]:
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
a = (d ** 2 + c1r ** 2 - c2r ** 2) / (2 * d)  # distance from c1c to intersecting chord
b = sqrt(c1r ** 2 - a ** 2)  # half-length of the intersecting chord

# Construct a basis
AB = subtract((c2x, c2y), (c1x, c1y))  # AB = B - A
e1 = AB / d  # e1 = AB / |AB|; norm(e1) = 1
assert round(norm(e1), ndigits=5) == 1, "Error: unit vector e1 = AB/|AB| should be 1"
e2 = array([[0, -1], [1, 0]]) @ e1  # Applies rotation matrix perpendicular to e1

possible_ip = []

# P_{1,2} = c1c + a·e1 +/- b·e2
for b_component in [b, -b]:
    ip = (c1x, c1y) + a * e1 + b_component * e2
    possible_ip.append(ip)

# Then compare the [at most] two points and select the nearest to interior_centre
ip_dists = [dist(interior_centre, ip) for ip in possible_ip]
ip_n = array(ip_dists).argsort().argsort().tolist().index(0)
ip = possible_ip[ip_n]
plt.scatter(ip[0], ip[1], s=300, color="blue", alpha=0.5)
for ip in possible_ip:
    plt.scatter(ip[0], ip[1], s=30, color="red")

# for cc_n, (cc_xc, cc_yc) in enumerate(circle_centres_clockwise):
#     prev_xc, prev_yc = circle_centres_clockwise[cc_n - 1]
#     next_xc, next_yc = circle_centres_clockwise[(cc_n + 1) % len(circle_centres)]
#     # Each adjacent circle (2 per circle) may have 1 or 2 intersection points
#     prev_ip, prev_t = choose_interior_ip((cc_xc, cc_yc), (prev_xc, prev_yc))
#     next_ip, next_t = choose_interior_ip((cc_xc, cc_yc), (next_xc, next_yc))
#     # Assign radius based on whether in upper or lower half of the ellipse
#     if cc_yc > yc:
#         cc_r = lc_r
#     else:
#         cc_r = tc_r
#     if VISUALISE or SAVE_PLOT:
#         for arc_t in arange(prev_t, next_t, rad(10)):
#             arc_x = cc_xc + cc_r * cos(arc_t)
#             arc_y = cc_yc + cc_r * sin(arc_t)
#             plt.scatter(arc_x, arc_y, s=20, color="k")

if SAVE_PLOT:
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plt.savefig("circle-sketch.png", bbox_inches="tight")
elif VISUALISE:
    plt.show()
