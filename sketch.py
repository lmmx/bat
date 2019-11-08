import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, radians as rad, sqrt
from ellipse_intersect import intersection_points, intersection_t

VISUALISE = False
SAVE_PLOT = False

if VISUALISE or SAVE_PLOT:
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_aspect(1.0)
    plt.axis([0, 2000, 2000, 0])
    ax.set_title("Sketch of a bat from geometry of intersecting conics", size=18)
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)

scale_factor = 100

# On sketch diagram these are a:b ≈ 9:5
h_radius, v_radius = np.dot(scale_factor, [9, 5])
focal_len = sqrt(h_radius ** 2 - v_radius ** 2)

xc = 1000
yc = 1000

if VISUALISE or SAVE_PLOT:
    plt.scatter(xc, yc, s=10, color="k")

    arrow_opacity = 0.4

    # Draw the outer ellipse using measurements obtained by eye
    for alpha in np.arange(rad(0), rad(360), rad(3)):
        oe_x = xc + h_radius * cos(alpha)
        oe_y = yc + v_radius * sin(alpha)
        plt.scatter(oe_x, oe_y, s=2, color="k")

    # Leftward green arrow (a)
    plt.arrow(
        xc,
        yc,
        -1 * h_radius,
        0,
        length_includes_head=True,
        head_length=30,
        head_width=30,
        color="g",
        alpha=arrow_opacity,
    )

    # Upward green arrow (b)
    plt.arrow(
        xc,
        yc,
        0,
        -1 * v_radius,
        length_includes_head=True,
        head_length=30,
        head_width=30,
        color="g",
        alpha=arrow_opacity,
    )

    # Rightward purple arrow (c)
    plt.arrow(
        xc,
        yc,
        focal_len,
        0,
        length_includes_head=True,
        head_length=30,
        head_width=30,
        color="purple",
        alpha=arrow_opacity,
    )

    # Leftward yellow arrow (a/2)
    plt.arrow(
        xc,
        yc,
        -1 / 2 * h_radius,
        0,
        length_includes_head=True,
        head_length=30,
        head_width=30,
        color="yellow",
        alpha=arrow_opacity,
    )

    # Rightward blue arrow (a/2)
    plt.arrow(
        xc,
        yc,
        h_radius / 2,
        0,
        length_includes_head=True,
        head_length=30,
        head_width=30,
        color="b",
        alpha=arrow_opacity,
    )

# Determine top circle radii
# For now, just take it as 90% of v_radius
tc_scale = 0.9
tc_r = 0.5 * v_radius * tc_scale

# Store the circle centres, these will be used together later in the calculation
# of centre point adjacencies (i.e. which circles neighbour one another)

circle_centres = []

# Plot the middle top circle
mtc_dy = (1 - tc_scale) * (0.5 * v_radius)
mtc_xc = xc
mtc_yc = yc - (mtc_dy + v_radius / 2)
circle_centres.append((mtc_xc, mtc_yc))  # (CIRCLE 1)

if VISUALISE or SAVE_PLOT:
    # (CIRCLE 1)
    for alpha in np.arange(rad(0), rad(360), rad(1)):
        mtc_x = mtc_xc + tc_r * cos(alpha)
        mtc_y = mtc_yc + tc_r * sin(alpha)
        plt.scatter(mtc_x, mtc_y, s=6, color="pink")

# Calculate the intersection of:
# - the circular locus of tc_r radii around the horizontal radius midpoint
# - the elliptical locus of centres of circles touching the outer ellipse
# First for the left horizontal midpoint and then the right hand side one

# First make the circular locus of tc_r radii around the left midpoint
lmc_xc = xc - 1 / 2 * h_radius

if VISUALISE or SAVE_PLOT:
    for alpha in np.arange(rad(0), rad(360), rad(10)):
        lmc_x = lmc_xc + tc_r * cos(alpha)
        lmc_y = yc + tc_r * sin(alpha)
        plt.scatter(lmc_x, lmc_y, s=6, color="lime")

    # Now make the elliptical locus of circle centres along the outer ellipse
    for alpha in np.arange(rad(0), rad(360), rad(3)):
        el_x = xc + (h_radius - tc_r) * cos(alpha)
        el_y = yc + (v_radius - tc_r) * sin(alpha)
        plt.scatter(el_x, el_y, s=2, color="orange")

# After viewing this to confirm the proper value for tc_scale to obtain
# a point of intersection between the circular and elliptical loci, next
# calculate the point of intersection in the top left ellipse quadrant.
# I do this here by substituting the values of x and y from the parametric
# form of the equation for the circle into the Cartesian equation for the
# ellipse to get a (degree 4) polynomial in t whose real solutions give
# any points of intersection's (x,y) coordinates, obtained in the end
# by re-substituting t into the parametric equation of the circle.
# Here the array intersection_points provides the (x, y) coordinates,
# exported from `ellipse_intersect.py`.

if VISUALISE or SAVE_PLOT:
    pcol = ["red", "blue", "yellow", "magenta"]

    for ip_n, ip in enumerate(intersection_points):
        plt.scatter(ip[0], ip[1], s=50, color=pcol[ip_n])

ip_dists = [np.linalg.norm(ip - mtc_yc) for ip in intersection_points]
nearest_ip = intersection_points[ip_dists.index(np.min(ip_dists))]

# Plot the left top circle
ltc_xc = nearest_ip[0]
ltc_yc = nearest_ip[1]
circle_centres.append((ltc_xc, ltc_yc))  # (CIRCLE 2)

if VISUALISE or SAVE_PLOT:
    # (CIRCLE 2)
    for alpha in np.arange(rad(0), rad(360), rad(1)):
        ltc_x = ltc_xc + tc_r * cos(alpha)
        ltc_y = ltc_yc + tc_r * sin(alpha)
        plt.scatter(ltc_x, ltc_y, s=6, color="pink")

# Plot the right top circle
rtc_xc = (2 * mtc_xc) - ltc_xc  # Reflect across the vertical line x = ltc_xc
rtc_yc = ltc_yc
circle_centres.append((rtc_xc, rtc_yc))  # (CIRCLE 3)

if VISUALISE or SAVE_PLOT:
    # (CIRCLE 3)
    for alpha in np.arange(rad(0), rad(360), rad(1)):
        rtc_x = rtc_xc + tc_r * cos(alpha)
        rtc_y = rtc_yc + tc_r * sin(alpha)
        plt.scatter(rtc_x, rtc_y, s=6, color="pink")

# Now to draw the bottom half's circles, first find the leftmost intersection
# point of the left top circle and the horizontal bisecting line (f(x) = yc)
# Take same approach as in ellipse_intersect.py (but only cubic not quartic)
# The equation simplifies to "(Yc - y)*t^2 + (2*r)*t + (Yc - y) = 0"
hbt_p_2 = ltc_yc - yc
hbt_p_1 = 2 * tc_r
hbt_p_0 = ltc_yc - yc
hbt_p_coeff = [hbt_p_2, hbt_p_1, hbt_p_0]
hbt_roots = np.roots(hbt_p_coeff)
hbt_roots = [
    hbt_roots[ii].real
    for ii in np.sort(
        [[tt.real for tt in hbt_roots].index(t) for t in np.unique(np.real(hbt_roots))]
    )
]

hbt_ip = []
hbt_t = []

if VISUALISE or SAVE_PLOT:
    pcol = ["yellow", "palegreen", "magenta", "chocolate"]

for t_num, t in enumerate(hbt_roots):
    pol = [
        (c[1] * t ** c[0])
        for c in list(reversed(list(enumerate(reversed(hbt_p_coeff)))))
    ]
    if int(sum(pol)) != 0:
        print(
            f"Polynomial didn't evaluate to 0 as expected. "
            + f"Root {t} not valid. Skipping..."
        )
        continue  # Do not plot this root, skip to next root in hbt_roots (if any left)
    #
    x_crossing = ltc_xc + tc_r * ((1 - t ** 2) / (1 + t ** 2))
    y_crossing = ltc_yc + tc_r * ((2 * t) / (1 + t ** 2))
    #
    if VISUALISE or SAVE_PLOT:
        plt.scatter(x_crossing, y_crossing, s=50, color=pcol[t_num])
    #
    hbt_ip.append((x_crossing, y_crossing))
    hbt_t.append(t)

# Take the leftmost of the intersections (i.e. lowest x value)
hbt_ip_l = sorted(hbt_ip, key=lambda x: x[0])[0]
hbt_t_l = hbt_t[hbt_ip.index(hbt_ip_l)]

# Give the lower circles 60% the radius of the top ones and place the centre of the
# leftmost lower circle this distance away from the intersection point.
# Note that this means these 2 circles will be 'flat' to one another, not overlapping.
lower_circle_scale = 0.6  # Relative to the circles on the top half of the ellipse
lc_r = tc_r * lower_circle_scale  # Radius of the lower circle
llc_xc = hbt_ip_l[0] + (lc_r * ((1 - hbt_t_l ** 2) / (1 + hbt_t_l ** 2)))
llc_yc = hbt_ip_l[1] + (lc_r * ((2 * t) / (1 + t ** 2)))
circle_centres.append((llc_xc, llc_yc))  # (CIRCLE 4)

# Draw the first lower circle, the leftmost one
if VISUALISE or SAVE_PLOT:
    # (CIRCLE 4)
    for alpha in np.arange(rad(0), rad(360), rad(1)):
        llc_x = llc_xc + lc_r * cos(alpha)
        llc_y = llc_yc + lc_r * sin(alpha)
        plt.scatter(llc_x, llc_y, s=6, color="lightgreen")

# Next find the touching point of the lower circle which touches the central vertical
# bisecting line on its right hand side and 'lowest' (maximum y on our reversed y axis)
# centre without its edge (lc_r distance directly below it) crossing the outer ellipse
oem = (xc, yc + v_radius)

oem_x_crossing = xc - lc_r
oem_y_crossings = (
    np.array([-1, 1])
    * sqrt(v_radius ** 2 * (1 - ((oem_x_crossing - xc) ** 2 / (h_radius ** 2))))
    + yc
)
assert (
    len(oem_y_crossings[oem_y_crossings > yc]) == 1
), "There shouldn't be more than 2 points on the circumference of an ellipse at the same x value"
oem_y_crossing = oem_y_crossings[oem_y_crossings > yc].item()

plt.scatter(oem[0], oem[1], s=20, color="red")
plt.scatter(oem_x_crossing, oem_y_crossing, s=20, color="blue")

# Now draw this base midpoint left circle by settings its centre lc_r above this point
bmlc_xc = oem_x_crossing
bmlc_yc = oem_y_crossing - lc_r
circle_centres.append((bmlc_xc, bmlc_yc))  # (CIRCLE 5)

# (CIRCLE 5)
for alpha in np.arange(rad(0), rad(360), rad(1)):
    bmlc_x = bmlc_xc + lc_r * cos(alpha)
    bmlc_y = bmlc_yc + lc_r * sin(alpha)
    plt.scatter(bmlc_x, bmlc_y, s=6, color="lightgreen")

# Draw the final lower left quadrant circle centred at the midpoint of the other two
lc_dx, lc_dy = (bmlc_xc - llc_xc, bmlc_yc - llc_yc)
flmc_xc, flmc_yc = (llc_xc + lc_dx / 2, llc_yc + lc_dy / 2)
circle_centres.append((flmc_xc, flmc_yc))  # (CIRCLE 6)

plt.scatter(flmc_xc, flmc_yc, s=20, color="green")

# (CIRCLE 6)
for alpha in np.arange(rad(0), rad(360), rad(1)):
    flmc_x = flmc_xc + lc_r * cos(alpha)
    flmc_y = flmc_yc + lc_r * sin(alpha)
    plt.scatter(flmc_x, flmc_y, s=6, color="lightgreen")

# Lastly, mirror all these across the line f(y) = xc
# (CIRCLE 7)
r_llc_xc = 2 * xc - llc_xc
circle_centres.append((r_llc_xc, llc_yc))

# (CIRCLE 8)
r_bmlc_xc = 2 * xc - bmlc_xc
circle_centres.append((r_bmlc_xc, bmlc_yc))

# (CIRCLE 9)
r_flmc_xc = 2 * xc - flmc_xc
circle_centres.append((r_flmc_xc, flmc_yc))

if VISUALISE or SAVE_PLOT:
    # Reflected to top right circle in the lower right quadrant of the ellipse
    # (CIRCLE 7)
    for alpha in np.arange(rad(0), rad(360), rad(1)):
        r_llc_x = r_llc_xc + lc_r * cos(alpha)
        r_llc_y = llc_yc + lc_r * sin(alpha)
        plt.scatter(r_llc_x, r_llc_y, s=6, color="lightgreen")

    # Reflected to bottom left circle in lower right quadrant of the ellipse
    # (CIRCLE 8)
    for alpha in np.arange(rad(0), rad(360), rad(1)):
        r_bmlc_x = r_bmlc_xc + lc_r * cos(alpha)
        r_bmlc_y = bmlc_yc + lc_r * sin(alpha)
        plt.scatter(r_bmlc_x, r_bmlc_y, s=6, color="lightgreen")

    # Reflected to middle circle in lower right quadrant of the ellipse
    # (CIRCLE 9)
    for alpha in np.arange(rad(0), rad(360), rad(1)):
        r_flmc_x = r_flmc_xc + lc_r * cos(alpha)
        r_flmc_y = flmc_yc + lc_r * sin(alpha)
        plt.scatter(r_flmc_x, r_flmc_y, s=6, color="lightgreen")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# All circles have now been drawn and their interior arcs delineate the
# outline of the bat. Success! Next, to trace out this outline, first
# construct an adjacency matrix of the centre points

# The order in this array will match the sweep of t=0 to t=2*pi, which on our
# axes mean a relabelling of the circles numbered 1-9 as [0-based] sequence:
relabel_sequence = [7, 6, 8, 5, 3, 4, 0, 2, 1]
circle_centres_clockwise = [circle_centres[ri] for ri in relabel_sequence]

# An alternative way to do this is taking all-against-all distance transform of the
# matrix of circle centres, i.e. finding a permutation to relabel the circle centres
# according to proximity. This is useful even if it's possible to do by hand.

# Repeat the list of 9 circle_centres into a 9x9 matrix, each row has the same value
circle_centre_matrix = np.repeat(
    np.array(circle_centres), len(circle_centres), axis=0
).reshape(len(circle_centres), len(circle_centres), 2)
# The Euclidean distance between two points is sqrt(a^2 + b^2).
# A distance matrix calculates it for all the circle centres against all others,
dist_matrix = np.sqrt(
    np.sum(
        np.square(
            (np.subtract(circle_centre_matrix, np.swapaxes(circle_centre_matrix, 0, 1)))
        ),
        axis=2,
    )
)
assert np.all(dist_matrix.diagonal() == 0)  # Distance of a point to itself is zero
# Now find the 2 nearest points by finding the rank (ignoring 0th, which is the zero)
# The incidence matrix is True for the nearest and next-nearest circle centre
point_incidence_matrix = np.isin(dist_matrix.argsort().argsort(), (1, 2))
# nearest_pairs stores the indices of circle_centres corresponding to the 2 neighbours
nearest_pairs = np.where(point_incidence_matrix)[1].reshape(len(circle_centres), 2)
# for (n, (a,b)) in enumerate(nearest_pairs):
#    print(f"Circle centre {n+1} is nearest to {a+1} and {b+1}")
#
# Circle centre 1 is nearest to 2 and 3
# Circle centre 2 is nearest to 1 and 4
# Circle centre 3 is nearest to 1 and 7
# Circle centre 4 is nearest to 2 and 6
# Circle centre 5 is nearest to 6 and 8
# Circle centre 6 is nearest to 4 and 5
# Circle centre 7 is nearest to 3 and 9
# Circle centre 8 is nearest to 5 and 9
# Circle centre 9 is nearest to 7 and 8

# These are all correct, but I would like to start at the rightmost point and go around
# from 0 to 2*pi, the same way the circles are being plotted from rad(0) to rad(360)
oer = (xc + h_radius, yc)
# No need to square root to give Euclidean distance, since all that's needed is
# the smallest value's index: achieved here by finding index where the rank is 0
d_circ_oer = np.sum(np.square(np.subtract(oer, circle_centres)), axis=1)
starting_circle_index = np.where(d_circ_oer.argsort().argsort() == 0)[0].item()

# Now begin to build a permutation, initialised at this starting index, and add
# the nearest of the pairs each time until the permutation is of len(circle_centres)
circle_permutation = [starting_circle_index]
for i in np.arange(0, len(circle_centres) - 1):
    last_permuted = circle_permutation[-1]
    # Now find the nearest circle to this
    nearest_c = np.where(dist_matrix[last_permuted].argsort().argsort() == 1)[0].item()
    if nearest_c in circle_permutation:
        # This has already been encountered, so must use the other of the pair
        nearest_c = (set(nearest_pairs[last_permuted]) - set({nearest_c})).pop()
    circle_permutation.append(nearest_c)

assert len(circle_permutation) == len(circle_centres)
# Sucessfully recovered the relabelling permutation
relabelling_permutation = np.array(circle_permutation).argsort().tolist()
assert np.array_equal(relabelling_permutation, relabel_sequence)

if SAVE_PLOT:
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plt.savefig("sketch.png", bbox_inches="tight")
elif VISUALISE:
    plt.show()
