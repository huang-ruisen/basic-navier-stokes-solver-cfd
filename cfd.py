import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.animation import FuncAnimation
import random

global x
global y
x = np.arange(-101, 101, 1)
y = np.arange(-101, 101, 1)
x, y = np.meshgrid(x, y)

vx = np.zeros_like(x, dtype = float)
vy = np.zeros_like(y, dtype = float)
p = np.zeros_like(x, dtype = float)

vx_flow = int(input("enter vx added"))
vy_flow = int(input("enter vy added"))

vx = vx + vx_flow
vy = vy + vy_flow

r = 25

shape = Circle((0, 0), 25, facecolor = "red")

def intermediate_velocity(vx, vy, dt, nu, g, r):
    global x, y
    
    mask = x ** 2 + y ** 2 >= r ** 2

    
    vxdy, vxdx = np.gradient(vx)[0], np.gradient(vx)[1]
    vydy, vydx = np.gradient(vy)[0], np.gradient(vy)[1]
    
    vxd2x = np.gradient(vxdx)[1]
    vxd2y = np.gradient(vxdy)[0]
    vyd2x = np.gradient(vydx)[1]
    vyd2y = np.gradient(vydy)[0]
    
    convective_termx = vx * vxdx + vy * vxdy
    convective_termy = vx * vydx + vy * vydy
    
    temp1 = vxd2x + vxd2y
    temp2 = vyd2x + vyd2y
    viscocity_termx = nu * temp1
    viscocity_termy = nu * temp2
    
    temp1 = -convective_termx + viscocity_termx
    temp2 = -convective_termy + viscocity_termy - g
    
    vxn = vx + dt * temp1
    vyn = vy + dt * temp2
    
    vxn = np.where(mask, vxn, 0)
    vyn = np.where(mask, vyn, 0)
    return vxn, vyn
    
def pressure_possion(vxn, vyn, rho, dt, r):
    global p
    vxdx = np.gradient(vxn)[1]
    vydy = np.gradient(vyn)[0]
    mask = x ** 2 + y ** 2 >= r ** 2
    temp1 = vxdx + vydy
    temp2 = rho / dt
    b = temp1 * temp2
    
    error = 10e-6
    
    for k in range(1, 101):
        pn = np.copy(p)
        for i in range(1, 200):
            for j in range(1, 200):
                neighbourgridsum = pn[i, j + 1] + pn[i, j - 1] + pn[i + 1, j] + pn[i-1, j]
                temp3 = neighbourgridsum - b[i, j]
                p[i, j] = 0.25 * temp3
                
    p = np.where(mask, p, 0)
    
    return p

def correct_velocity(vxn, vyn, p, rho, dt):
    dpdy, dpdx = np.gradient(p, 1, 1)
    
    temp = dt / rho
    vx_corrected = vxn - temp * dpdx
    vy_corrected = vyn - temp * dpdy
    
    return vx_corrected, vy_corrected
     
     
fig, ax = plt.subplots()

speed = np.sqrt(vx ** 2 + vy ** 2)

r = 25
mask = x ** 2 + y ** 2 >= r ** 2

for frame in range(1, 6):
    vx[:, 0] = vx_flow
    vy[:, 0] = vy_flow
    
    vxn, vyn = intermediate_velocity(vx, vy, 0.01, 1.6e-5, 0, r)
    p = pressure_possion(vxn, vyn, 1, 0.01, r)
    vx_corrected, vy_corrected = correct_velocity(vxn, vyn, p, 1, 0.01)
    
    vx_corrected = np.where(mask, vx_corrected, 0)
    vy_corrected = np.where(mask, vy_corrected, 0)
    
    vx = vx_corrected
    vy = vy_corrected
    
plt.streamplot(x, y, vx, vy, density = 5, color = speed, cmap = "viridis")
plt.colorbar()
ax.add_patch(shape)
plt.show()
