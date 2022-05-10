
from __future__ import (print_function, division)
import time

import numpy as np
import matplotlib.pyplot as plt

experiment = '1d'
plot_interval = 20          # строить каждые n шагов


nx = 128
ny = 129

H  = 100.0          # глубина
Lx = 2.0e7          # масштаб по x
Ly = 1.0e7          # масштаб по y

boundary_condition = 'walls'

f0 = 0 #f - кориолис, f = f0 + beta y
beta =  0.0
g = 1.0 #g - гравитация


nu = 5.0e4 #nu - к-т вязкости,
r = 1.0e-4 #трение тип

dt = 1000.0  #шаг по времени
dx = Lx / nx
dy = Ly / ny

# граничные:
_u = np.zeros((nx+3, ny+2))
_v = np.zeros((nx+2, ny+3))
_h = np.zeros((nx+2, ny+2))

# текущие:
u = _u[1:-1, 1:-1]               # (nx+1, ny)
v = _v[1:-1, 1:-1]               # (nx, ny+1)
h = _h[1:-1, 1:-1]               # (nx, ny)

state = np.array([u, v, h])

ux = (-Lx/2 + np.arange(nx+1)*dx)[:, np.newaxis]
vx = (-Lx/2 + dx/2.0 + np.arange(nx)*dx)[:, np.newaxis]

vy = (-Ly/2 + np.arange(ny+1)*dy)[np.newaxis, :]
uy = (-Ly/2 + dy/2.0 + np.arange(ny)*dy)[np.newaxis, :]

hx = vx
hy = uy

t = 0.0                 # время с начала мультика
tc = 0                  # кол-во шагов


#вспомогательные функции

def update_boundaries():
    """if boundary_condition == 'periodic':
            _u[0, :] = _u[-3, :]
            _u[1, :] = _u[-2, :]
            _u[-1, :] = _u[2, :]
            _v[0, :] = _v[-2, :]
            _v[-1, :] = _v[1, :]
            _h[0, :] = _h[-2, :]
            _h[-1, :] = _h[1, :]



        if boundary_condition == 'walls':"""
    _u[0, :] = 0
    _u[1, :] = 0
    _u[-1, :] = 0
    _u[-2, :] = 0


    _v[0, :] = _v[1, :]
    _v[-1, :] = _v[-2, :]
    _h[0, :] = _h[1, :]
    _h[-1, :] = _h[-2, :]


    for field in state:
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]


        field[0, 0] =  0.5*(field[1, 0] + field[0, 1])
        field[-1, 0] = 0.5*(field[-2, 0] + field[-1, 1])
        field[0, -1] = 0.5*(field[1, -1] + field[0, -2])
        field[-1, -1] = 0.5*(field[-1, -2] + field[-2, -1])


def diffx(psi): #∂/∂x[psi][i,j] = (psi[i+1/2, j] - psi[i-1/2, j]) / dx
    global dx
    return (psi[1:,:] - psi[:-1,:]) / dx

def diff2x(psi): #∂2/∂x2[psi][i,j] = (psi[i+1, j] - psi[i, j] + psi[i-1, j]) / dx^2
    global dx
    return (psi[:-2, :] - 2*psi[1:-1, :] + psi[2:, :]) / dx**2

def diff2y(psi): #∂2/∂y2[psi][i,j] = (psi[i, j+1] - psi[i, j] + psi[i, j-1]) / dy^2
    global dy
    return (psi[:, :-2] - 2*psi[:, 1:-1] + psi[:, 2:]) / dy**2

def diffy(psi): # ∂/∂y[psi][i,j] = (psi[i, j+1/2] - psi[i, j-1/2]) / dy
    global dy
    return (psi[:, 1:] - psi[:,:-1]) / dy

def centre_average(phi): #среднее значение по четырем точкам в центрах между точками сетки
    return 0.25*(phi[:-1,:-1] + phi[:-1,1:] + phi[1:, :-1] + phi[1:,1:])

def y_average(phi): # среднее по y
    return 0.5*(phi[:,:-1] + phi[:,1:])

def x_average(phi): # среднее по x
    return 0.5*(phi[:-1,:] + phi[1:,:])

def divergence():
    return diffx(u) + diffy(v)

def del2(phi): # лапласиан
    return diff2x(phi)[:, 1:-1] + diff2y(phi)[1:-1, :]

def uvatuv():
    global _u, _v
    ubar = centre_average(_u)[1:-1, :]
    vbar = centre_average(_v)[:, 1:-1]
    return ubar, vbar

def uvath():
    global u, v
    ubar = x_average(u)
    vbar = y_average(v)
    return ubar, vbar

def absmax(psi):
    return np.max(np.abs(psi))


#динамика
def forcing(): #добавить внешние (du, dv, dh)
    global u, v, h
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    dh = np.zeros_like(h)
    return np.array([du, dv, dh])

sponge_ny = ny//7
sponge = np.exp(-np.linspace(0, 5, sponge_ny))

def damping(var): #ну это про трение
    global sponge, sponge_ny
    var_sponge = np.zeros_like(var)
    var_sponge[:, :sponge_ny] = sponge[np.newaxis, :]
    var_sponge[:, -sponge_ny:] = sponge[np.newaxis, ::-1]
    return var_sponge*var

def rhs(): #вычисляет правую часть уравнений на u, v, h
    u_at_v, v_at_u = uvatuv()   # (nx, ny+1), (nx+1, ny)

    # уравнение на h
    h_rhs = -H*divergence() + nu*del2(_h) - r*damping(h)

    # уравнение на u
    dhdx = diffx(_h)[:, 1:-1]  # (nx+1, ny)
    u_rhs = (f0 + beta*uy)*v_at_u - g*dhdx + nu*del2(_u) - r*damping(u)

    # уравнение на v
    dhdy  = diffy(_h)[1:-1, :]   # (nx, ny+1)
    v_rhs = -(f0 + beta*vy)*u_at_v - g*dhdy + nu*del2(_v) - r*damping(v)

    return np.array([u_rhs, v_rhs, h_rhs]) + forcing()

_ppdstate, _pdstate = 0,0

def step():
    global dt, t, tc, _ppdstate, _pdstate
    update_boundaries()
    dstate = rhs()
    # по времени метод Адамса — Башфорта
    if tc==0:
        dt1 = dt
        dt2 = 0.0
        dt3 = 0.0
    elif tc==1:
        dt1 = 1.5*dt
        dt2 = -0.5*dt
        dt3 = 0.0
    else:
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt

    newstate = state + dt1*dstate + dt2*_pdstate + dt3*_ppdstate
    u[:], v[:], h[:] = newstate
    _ppdstate = _pdstate
    _pdstate = dstate

    t  += dt
    tc += 1


# начальные условия
"""if experiment == '2d':
    # create a single disturbance in the domain:
    # a gaussian at position gx, gy, with radius gr
    gx =  2.0e6
    gy =  0.0
    gr =  2.0e5
    h0 = np.exp(-((hx - gx)**2 + (hy - gy)**2)/(2*gr**2))*H*0.01
    u0 = u * 0.0
    v0 = v * 0.0

if experiment == '1d':"""
gx =  2.0e6
gy =  0.0
gr =  2.0e5
h0 = np.exp(-((hx - gx)**2 + (hy - gy)**2)/(2*gr**2))*H*0.01
#h0 = -np.tanh(100*hx/Lx)
v0 = v * 0.0
u0 = u * 0.0
r = 0.0

# можно переменные поля в начальные условия....
u[:] = u0
v[:] = v0
h[:] = h0

# графики
plt.ion()                         # обновляяяются
fig = plt.figure(figsize=(8*Lx/Ly, 8))

nc = 12
colorlevels = np.concatenate([np.linspace(-1, -.05, nc), np.linspace(.05, 1, nc)])


def plot(u, v, h):
        plt.clf()

        h0max = absmax(h0)
        plt.subplot(121)
        plt.plot(hx, h[:, ny//2], 'b', linewidth=2)
        plt.ylabel('height')
        plt.ylim(-h0max*0.5, h0max)



        plt.subplot(122)
        plt.plot(ux, u[:, ny//2], linewidth=2)
        plt.xlabel('x',size=16)
        plt.ylabel('u velocity')
        plt.ylim(-h0max*.05, h0max*.05)

        plt.pause(0.001)
        plt.draw()

        j = i + 1
        plt.savefig('mult_2_fig_%d' % j)


#запуууууск
nsteps = 3000
for i in range(nsteps):
    step()
    if i % plot_interval == 0:
        plot(*state)
        print('[t={:7.2f} u: [{:.3f}, {:.3f}], v: [{:.3f}, {:.3f}], h: [{:.3f}, {:.2f}]'.format(
            t/86400,
            u.min(), u.max(),
            v.min(), v.max(),
            h.min(), h.max()))

