import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, radians as rad, sqrt
from ellipse_intersect import intersection_points, intersection_t

ax = plt.subplot()
ax.set_aspect(1.0)

scale_factor = 100

# On sketch diagram these are a:b ≈ 9:5
h_radius, v_radius = np.dot(scale_factor, [9, 5])
focal_len = sqrt(h_radius ** 2 - v_radius ** 2)

xc = 1000
yc = 1000

plt.axis([0, 2000, 2000, 0])

plt.scatter(xc, yc, s=10, color="k")

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
)

# Determine top circle radii
# For now, just take it as 90% of v_radius
tc_scale = 0.9
tc_r = 0.5 * v_radius * tc_scale

# Plot the middle top circle
mtc_dy = (1 - tc_scale) * (0.5 * v_radius)
mtc_xc = xc
mtc_yc = yc - (mtc_dy + v_radius / 2)

for alpha in np.arange(rad(0), rad(360), rad(10)):
    mtc_x = mtc_xc + tc_r * cos(alpha)
    mtc_y = mtc_yc + tc_r * sin(alpha)
    plt.scatter(mtc_x, mtc_y, s=6, color="pink")

# Calculate the intersection of:
# - the circular locus of tc_r radii around the horizontal radius midpoint
# - the elliptical locus of centres of circles touching the outer ellipse
# First for the left horizontal midpoint and then the right hand side one

# First make the circular locus of tc_r radii around the left midpoint
lmc_xc = xc - 1 / 2 * h_radius
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
# any points of intersection's (x,y) coordinates.
#
# ( (xc + tc_r*(1-t**2)**2/(1+t**2)**2) / a**2) + ( (yc + tc_r*(4*t**2/(1+t**2)**2)) / b**2) - 1

# (Cheat for now and use Sympy geometry class intersection method)

# Plot the t values derived from parametric equation for ellipse
# as the alpha values into the equation for the ellipse
pcol = ["red","blue", "yellow"]
for ip in enumerate(intersection_t):
    alpha_x = ip[1][0] # use the arccos derived value
    alpha_y = ip[1][1] # use the arcsin derived value
    #alpha = 1.3595854464357182
    #alpha = 4.92355998607438685
    el_x = xc + (h_radius - tc_r) * cos(alpha_x)
    el_y = yc + (v_radius - tc_r) * sin(alpha_y)
    plt.scatter(el_x, el_y, s=50, color=pcol[ip[0]])
