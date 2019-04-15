# TODO: Figure out if I need more libs?
import numpy as np


# Code interpreted from Andreas matlab code

# Grid in fm
rmax = 10.0

# Amount of steps
N = 10000

# Step length
h = rmax/N

# Printing step length
print(f'Step length h: {h}')

# Initializing grid and the potential V(r)
# For every r. OBS: Do not use rstart = 0!

# TODO: Figure out how to represent negative exponential in a correct way
rstart = 1.0 ** (-1)
r = np.arange((1/10), rmax, h)

print(f'Rang r is: {r}')

