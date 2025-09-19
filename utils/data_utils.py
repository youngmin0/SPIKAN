from functools import partial

import jax
import jax.numpy as jnp


# 3d time-dependent navier-stokes forcing term
def navier_stokes4d_forcing_term(t, x, y, z, nu):
    # forcing terms in the PDE
    # f_x = -24*jnp.exp(-18*nu*t)*jnp.sin(2*y)*jnp.cos(2*y)*jnp.sin(z)*jnp.cos(z)
    f_x = -6*jnp.exp(-18*nu*t)*jnp.sin(4*y)*jnp.sin(2*z)
    # f_y = -24*jnp.exp(-18*nu*t)*jnp.sin(2*x)*jnp.cos(2*x)*jnp.sin(z)*jnp.cos(z)
    f_y = -6*jnp.exp(-18*nu*t)*jnp.sin(4*x)*jnp.sin(2*z)
    # f_z = 24*jnp.exp(-18*nu*t)*jnp.sin(2*x)*jnp.cos(2*x)*jnp.sin(2*y)*jnp.cos(2*y)
    f_z = 6*jnp.exp(-18*nu*t)*jnp.sin(4*x)*jnp.sin(4*y)
    return f_x, f_y, f_z


# 3d time-dependent navier-stokes exact vorticity
def navier_stokes4d_exact_w(t, x, y, z, nu):
    # analytic form of vorticity
    w_x = -3*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.cos(2*y)*jnp.cos(z)
    w_y = 6*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.sin(2*y)*jnp.cos(z)
    w_z = -6*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.cos(2*y)*jnp.sin(z)
    return w_x, w_y, w_z


# 3d time-dependent navier-stokes exact velocity
def navier_stokes4d_exact_u(t, x, y, z, nu):
    # analytic form of velocity
    u_x = 2*jnp.exp(-9*nu*t)*jnp.cos(2*x)*jnp.sin(2*y)*jnp.sin(z)
    u_y = -1*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.cos(2*y)*jnp.sin(z)
    u_z = -2*jnp.exp(-9*nu*t)*jnp.sin(2*x)*jnp.sin(2*y)*jnp.cos(z)
    return u_x, u_y, u_z


# 3d time-dependent flow-mixing exact solution
def flow_mixing3d_exact_u(t, x, y, omega):
    return -jnp.tanh((y/2)*jnp.cos(omega*t) - (x/2)*jnp.sin(omega*t))


# 3d time-dependent flow-mixing parameters
def flow_mixing3d_params(t, x, y, v_max, require_ab=False):
    # t, x, y must be meshgrid
    r = jnp.sqrt(x**2 + y**2)
    v_t = ((1/jnp.cosh(r))**2) * jnp.tanh(r)
    omega = (1/r)*(v_t/v_max)
    a, b = None, None
    if require_ab:
        a = -(v_t/v_max)*(y/r)
        b = (v_t/v_max)*(x/r)
    return omega, a, b

@jax.jit
def taylor_couette_2d_exact_u(r, R1, R2, Omega1, Omega2):
    A = (Omega2 * R2**2 - Omega1 * R1**2) / (R2**2 - R1**2)
    B = (Omega1 - Omega2) * R1**2 * R2**2 / (R2**2 - R1**2)
    u_theta = A * r + B / r
    
    return u_theta