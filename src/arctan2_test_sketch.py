import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, arccos, arctan2, array, array_equal, dot, subtract
from numpy.linalg import norm
from math import sin, cos, radians as rad, sqrt

VISUALISE = False
SAVE_PLOT = True

if VISUALISE:
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_aspect(1.0)
    # ASCENDING Y AXIS IN THIS PLOT
    plt.axis([-4, 4, -4, 4])
    ax.set_title("Recovering points from angles by arctan2", size=18)
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)

xc = 0
yc = 0

if VISUALISE:
    plt.scatter(xc, yc, s=10, color="k")

r = sqrt(8)
circle_centre = (0, 0)

# Plot the circle

if VISUALISE or SAVE_PLOT:
    # CIRCLE i
    cxc, cyc = circle_centre
    for alpha in arange(rad(0), rad(360), rad(1)):
        c_x = cxc + r * cos(alpha)
        c_y = cyc + r * sin(alpha)
        plt.scatter(c_x, c_y, s=6, color="pink")


def plot_ip(ip, circle_centre=circle_centre):
    cxc, cyc = circle_centre
    ipx, ipy = ip
    t = arctan2(ipy - cxc, ipx - cyc)
    if t < 0:
        t += 2 * np.pi
    print(f"{ip}: {t}")
    x = cxc + r * cos(t)
    y = cxc + r * sin(t)
    return (x, y)

ip_n = (2, -2)
plt.scatter(ip_n[0], ip_n[1], s=300, color="blue", alpha=0.5)

ip_e = (2, 2)
plt.scatter(ip_e[0], ip_e[1], s=300, color="red", alpha=0.5)

ip_s = (-2, 2)
plt.scatter(ip_s[0], ip_s[1], s=300, color="yellow", alpha=0.5)

ip_w = (-2, -2)
plt.scatter(ip_w[0], ip_w[1], s=300, color="green", alpha=0.5)

pn = plot_ip(ip_n)
plt.scatter(pn[0], pn[1], s=40, color="blue", alpha=0.9)
pe = plot_ip(ip_e)
plt.scatter(pe[0], pe[1], s=40, color="red", alpha=0.9)
ps = plot_ip(ip_s)
plt.scatter(ps[0], ps[1], s=40, color="yellow", alpha=0.9)
pw = plot_ip(ip_w)
plt.scatter(pw[0], pw[1], s=40, color="green", alpha=0.9)

if SAVE_PLOT:
    plt.savefig("arctan2_demo.png", bbox_inches="tight")
elif VISUALISE:
    plt.show()
