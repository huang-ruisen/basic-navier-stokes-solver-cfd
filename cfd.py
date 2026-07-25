import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation
import random

dx = 1
dy = 1
x = np.arange(-101, 101, dx)
y = np.arange(-101, 101, dy)
x, y = np.meshgrid(x, y)

vx = np.zeros_like(x, dtype = float)
vy = np.zeros_like(y, dtype = float)
p = np.zeros_like(x, dtype = float)

vx_flow = int(input("enter vx added"))
vy_flow = int(input("enter vy added"))

vx = vx + vx_flow
vy = vy + vy_flow

sumvxvyerr = abs(vx_flow) + abs(vy_flow) + 10e-5
dtunadjusted = dx / sumvxvyerr
dt = dtunadjusted * 10e-3

r = 25

shape = Circle((0, 0), 25, facecolor = "red")

def intermediate_velocity(vx, vy, dt, nu, g, r):
    global x, y
   
    mask = x ** 2 + y ** 2 >= r ** 2 
    
    vxdy, vxdx = np.gradient(vx, dy, dx)[0], np.gradient(vx, dy, dx)[1]
    vydy, vydx = np.gradient(vy, dy, dx)[0], np.gradient(vy, dy, dx)[1]
    
    vxd2x = np.gradient(vxdx, dy, dx)[1]
    vxd2y = np.gradient(vxdy, dy, dx)[0]
    vyd2x = np.gradient(vydx, dy, dx)[1]
    vyd2y = np.gradient(vydy, dy, dx)[0]
    
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
    vxdx = np.gradient(vxn, dy, dx)[1]
    vydy = np.gradient(vyn, dy, dx)[0]
    dpdy, dpdx = np.gradient(p, dy, dx)
    mask = x ** 2 + y ** 2 >= r ** 2
    nx = x / np.sqrt(x ** 2 + y ** 2)
    ny = y / np.sqrt(x ** 2 + y ** 2)
    dpdn = dpdx * nx + dpdy * ny
    temp1 = vxdx + vydy
    temp2 = rho / dt
    b = temp1 * temp2
    
    error = 10e-6
    
    for k in range(1, 551):
        pn = np.copy(p)
        p = np.where(mask, p, p + dpdn * dx)
        neighbourgridsum = pn[2:, 1:-1] + pn[:-2, 1:-1] + pn[1:-1, :-2] + pn[1:-1, 2:]
        temp3 = b[1:-1, 1:-1] * dx ** 2
        temp4 = neighbourgridsum - temp3
        p[1:-1, 1:-1] = 0.25 * temp4
    
    return p

def correct_velocity(vxn, vyn, p, rho, dt):
    dpdy, dpdx = np.gradient(p, dy, dx)
    
    temp = dt / rho
    vx_corrected = vxn - temp * dpdx
    vy_corrected = vyn - temp * dpdy
    
    return vx_corrected, vy_corrected
     
     
fig, ax = plt.subplots()

r = 25
mask = x ** 2 + y ** 2 >= r ** 2

vx[:, 0] = vx_flow
vy[:, 0] = vy_flow

for f in range(1, 51):
    
    vxn, vyn = intermediate_velocity(vx, vy, dt, 1.6e-5, 0, r)
    p = pressure_possion(vxn, vyn, 1.2, dt, r)
    vx_corrected, vy_corrected = correct_velocity(vxn, vyn, p, 1.2, dt)
    
    vx_corrected = np.where(mask, vx_corrected, 0)
    vy_corrected = np.where(mask, vy_corrected, 0)
    vx = vx_corrected
    vy = vy_corrected

speed = np.sqrt(vx ** 2 + vy ** 2)

theta = np.linspace(0, 2 * np.pi, 360)
x_surface = r * np.cos(theta)
y_surface = r * np.sin(theta)
vxdy, vxdx = np.gradient(vx, dy, axis = 0), np.gradient(vx, dx, axis = 1)
vydy, vydx = np.gradient(vy, dy, axis = 0), np.gradient(vy, dx, axis = 1)
surface_point = np.vstack((y_surface, x_surface)).T

xc = np.arange(-101, 101, dx)
yc = np.arange(-101, 101, dy)

dvxdx = RegularGridInterpolator((yc, xc), vxdx, bounds_error = False, fill_value = 0)(surface_point)
dvxdy = RegularGridInterpolator((yc, xc), vxdy, bounds_error = False, fill_value = 0)(surface_point)
dvydx = RegularGridInterpolator((yc, xc), vydx, bounds_error = False, fill_value = 0)(surface_point)
dvydy = RegularGridInterpolator((yc, xc), vydy, bounds_error = False, fill_value = 0)(surface_point)

temp1 = np.sin(theta) * np.cos(theta)
temp2 = np.sin(theta) ** 2 * np.cos(theta) ** 2
temp3 = dvydy - dvxdx
temp4 = dvxdy + dvydx

shear_normal = 2 * temp3 * temp1
shear_tangent = temp4 * temp2

nu = 1.6e-5
rho = 1.2
viscocity = nu * rho

temp5 = shear_normal + shear_tangent

shear_stress = viscocity * temp5

p_surface = dvxdy = RegularGridInterpolator((yc, xc), p, bounds_error = False, fill_value = 0)(surface_point)

dthet =  2 * np.pi / 360
da = r * dthet
frds = np.sum(shear_stress * dthet * np.sin(theta))
fods = np.sum(p_surface * da * np.cos(theta))

d = 2 * r

temp6 = 0.5 * rho * vx_flow ** 2 * d

cd_friction = frds / temp6
cd_form = fods / temp6

cd_total = cd_friction + cd_form

total_drag = 0.5 * cd_total * vx_flow ** 2 * rho * d

print(cd_friction)
print(cd_form)
print(total_drag)
print(cd_total)

plt.streamplot(x, y, vx, vy, density = 2, color = speed, cmap = "viridis")
plt.colorbar()
ax.add_patch(shape)
plt.show()
