import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, radians as rad, sqrt

fig, ax = plt.subplots()

ax.set_aspect(1.0)

r = 40
xc = 50
yc = 50

# Leave the same amount of space either side
plt.axis([0, 100, 100, 0])

plt.scatter(xc, yc, s=10, color="k")

for alpha in np.arange(rad(0), rad(360), rad(2)):
    x = xc + r * cos(alpha)
    y = yc + r * sin(alpha)
    plt.scatter(x, y, s=5, color="k")
