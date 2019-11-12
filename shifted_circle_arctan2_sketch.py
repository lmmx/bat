import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, arccos, arctan2, array, array_equal, dot, subtract
from numpy.linalg import norm
from math import sin, cos, radians as rad, sqrt

VISUALISE = True

if VISUALISE:
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_aspect(1.0)
    # ASCENDING Y AXIS IN THIS PLOT
    plt.axis([12.5, 15.5, 10.5, 13.5])
    ax.set_title("Recovering points from angles by arctan2", size=18)
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)

r = 1.35
circle_centre = (14, 12)

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
    t = arctan2(ipy - cyc, ipx - cxc)
    if t < 0:
        t += 2 * np.pi
    assert not t > 2 * np.pi, "Unexpected value from arctan2: greater than 2*pi"
    x = cxc + r * cos(t)
    y = cyc + r * sin(t)
    print(f"{ip} ⇒ {(x,y)}: {t} (= {t * (360 / (2*np.pi))}°)")
    return (x, y)


test1 = (13.23103016, 10.89041206)
test2 = (13.0, 11.09308214)
plt.scatter(test1[0], test1[1], color="k", alpha=0.5)
plt.scatter(test2[0], test2[1], color="k", alpha=0.5)
p1 = plot_ip(test1, circle_centre=circle_centre)
p2 = plot_ip(test2, circle_centre=circle_centre)
plt.scatter(p1[0], p1[1], s=40, color="blue")
plt.scatter(p2[0], p2[1], s=40, color="red")

if VISUALISE:
    plt.show()
