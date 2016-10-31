# Gray-Scott model
# Would be good exercise to parallelize this script

import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow, show, gray, hot, draw
import time

L = 2.5
m, n = 255, 255
lamda = 2e-5
mu = 10**-5
dt = 1.
kappa = 0.06
F = 0.038
tsteps = 1000
dx = L/m  # x step
dy = L/n  # y step

start_time = time.time()

# initialize concentrations to steady state (u,v) = (1,0)
# third axis of the matrices is time: 0 for previous step, 1 for current step
U = np.zeros((m+1,n+1,2), float) + 1 # m+1 so I don't have to write m-1 everywhere
V = np.zeros((m+1,n+1,2), float)

# initialize 20x20 center square to (u,v) = (0.5, 0.25)
centerx = int(m)//2
centery = int(n)//2


for i in range(centerx-10, centerx+10):
	for j in range(centery-10, centery+10):
		U[i,j,1] = 0.5
		V[i,j,1] = 0.25

# slight rustle of all gridpoints by no more than 1%
for i in range(m-1):
	for j in range(n-1):
		U[i,j,1] += U[i,j,1] * np.random.uniform(-.01, 0.01)
		if U[i,j,1] > 1:
			U[i,j,1] = 1
		V[i,j,1] += V[i,j,1] * np.random.uniform(-.01, 0.01)

# initialize boundary conditions
U[m,:,1]   = U[m-2,:,1]  # periodic boundary for i=0
U[m-1,:,1] = U[0,:,1]    # periodic boundary for i=m-2
U[:,n,1]   = U[:,n-2,1]  # periodic boundary for j=0
U[:,n-1,1] = U[:,0,1]    # periodic boundary for j=n-2
U[m-1,n-1,1] = U[0,0,1]  # periodic corner for i=m,j=n
U[m,n,1] = U[m-2,n-2,1]  # periodic corner for i,j=0

V[m,:,1]   = V[m-2,:,1]  # periodic boundary for i=0
V[m-1,:,1] = V[0,:,1]    # periodic boundary for i=m-2
V[:,n,1]   = V[:,n-2,1]  # periodic boundary for j=0
V[:,n-1,1] = V[:,0,1]    # periodic boundary for j=n-2
V[m-1,n-1,1] = V[0,0,1]  # periodic corner for i=m,j=n
V[m,n,1] = V[m-2,n-2,1]  # periodic corner for i,j=0

imshow(U[:m-2, :n-2, 1])
gray()
#hot()
show()


# carry out the simulation!

for t in range(1, tsteps):
	U[:,:,0] = U[:,:,1]
	V[:,:,0] = V[:,:,1]

	for i in range(m-1):
		for j in range(n-1):

			# use forward Euler method applied to PDEs 
			U[i,j,1] = U[i,j,0] + dt*( (lamda/dx**2) * (U[i+1,j,0] - 2*U[i,j,0] + U[i-1,j,0]) + \
				  (lamda / dy**2) * (U[i,j+1,0] - 2*U[i,j,0] + U[i,j-1,0]) + F*(1-U[i,j,0]) - \
				  U[i,j,0]*V[i,j,0]**2 )

			V[i,j,1] = V[i,j,0] + dt*( (mu/dx**2) * (V[i+1,j,0] - 2*V[i,j,0] + V[i-1,j,0]) + \
				  (mu / dy**2) * (V[i,j+1,0] - 2*V[i,j,0] + V[i,j-1,0]) - (F+kappa)*V[i,j,0] + \
				  U[i,j,0]*V[i,j,0]**2 )

	U[m,:,1]   = U[m-2,:,1]  # periodic boundary for i=0
	U[m-1,:,1] = U[0,:,1]    # periodic boundary for i=m-2
	U[:,n,1]   = U[:,n-2,1]  # periodic boundary for j=0
	U[:,n-1,1] = U[:,0,1]    # periodic boundary for j=n-2
	U[m-1,n-1,1] = U[0,0,1]  # periodic corner for i=m,j=n
	U[m,n,1] = U[m-2,n-2,1]  # periodic corner for i,j=0

	V[m,:,1]   = V[m-2,:,1]  # periodic boundary for i=0
	V[m-1,:,1] = V[0,:,1]    # periodic boundary for i=m-2
	V[:,n,1]   = V[:,n-2,1]  # periodic boundary for j=0
	V[:,n-1,1] = V[:,0,1]    # periodic boundary for j=n-2
	V[m-1,n-1,1] = V[0,0,1]  # periodic corner for i=m,j=n
	V[m,n,1] = V[m-2,n-2,1]  # periodic corner for i,j=0



end_time = time.time() - start_time
print(end_time)

imshow(U[:m-2, :n-2, 1])
gray()
#hot()
show()
