import os
from utils.data_utils import *

import jax
import scipy.io


#======================== Navier-Stokes equation 3-d ========================#
#---------------------------------- SPINN, SPIKAN -----------------------------------#
def _spinn_train_generator_navier_stokes3d(nt, nxy, data_dir, result_dir, marching_steps, step_idx, offset_num, key):
    keys = jax.random.split(key, 2)
    gt_data = scipy.io.loadmat(os.path.join(data_dir, 'w_data.mat'))
    t = gt_data['t']

    # initial points
    ti = jnp.zeros((1, 1))
    xi = gt_data['x']
    yi = gt_data['y']
    if step_idx == 0:
        # get data from ground truth
        w0 = gt_data
    else:
        # get data from previous time window prediction
        w0 = scipy.io.loadmat(os.path.join(result_dir, '..', f'IC_pred/w0_{step_idx}.mat'))
        ti = w0['t']

    # collocation points
    tc = jnp.expand_dims(jnp.linspace(start=0., stop=t[0][-1], num=nt, endpoint=False), axis=1)
    xc = jnp.expand_dims(jnp.linspace(start=0., stop=2.*jnp.pi, num=nxy, endpoint=False), axis=1)
    yc = jnp.expand_dims(jnp.linspace(start=0., stop=2.*jnp.pi, num=nxy, endpoint=False), axis=1)

    if marching_steps != 0:
        # when using time marching
        Dt = t[0][-1] / marching_steps  # interval of a single time window
        # generate temporal coordinates within current time window
        if step_idx == 0:
            tc = jnp.expand_dims(jnp.linspace(start=0., stop=Dt*(step_idx+1), num=nt, endpoint=False), axis=1)
        else:
            tc = jnp.expand_dims(jnp.linspace(start=w0['t'][0][0], stop=Dt*(step_idx+1), num=nt, endpoint=False), axis=1)

    # for stacking multi-input grid
    tc_mult = jnp.expand_dims(tc, axis=0)
    xc_mult = jnp.expand_dims(xc, axis=0)
    yc_mult = jnp.expand_dims(yc, axis=0)

    # maximum value of offsets
    dt = tc[1][0] - tc[0][0]
    dxy = xc[1][0] - xc[0][0]

    # create offset values (zero is included by default)
    offset_t = jax.random.uniform(keys[0], (offset_num-1,), minval=0., maxval=dt)
    offset_xy = jax.random.uniform(keys[1], (offset_num-1,), minval=0., maxval=dxy)

    # make multi-grid
    for i in range(offset_num-1):
        tc_mult = jnp.concatenate((tc_mult, jnp.expand_dims(tc + offset_t[i], axis= 0)), axis=0)
        xc_mult = jnp.concatenate((xc_mult, jnp.expand_dims(xc + offset_xy[i], axis=0)), axis=0)
        yc_mult = jnp.concatenate((yc_mult, jnp.expand_dims(yc + offset_xy[i], axis=0)), axis=0)

    return tc_mult, xc_mult, yc_mult, ti, xi, yi, w0['w0'], w0['u0'], w0['v0']


#======================== Navier-Stokes equation 4-d ========================#
#---------------------------------- SPINN, SPIKAN -----------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _spinn_train_generator_navier_stokes4d(nc, nu, key):
    keys = jax.random.split(key, 4)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0., maxval=5.)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=0., maxval=2.*jnp.pi)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=0., maxval=2.*jnp.pi)
    zc = jax.random.uniform(keys[3], (nc, 1), minval=0., maxval=2.*jnp.pi)

    tcm, xcm, ycm, zcm = jnp.meshgrid(
        tc.ravel(), xc.ravel(), yc.ravel(), zc.ravel(), indexing='ij'
    )
    fc = navier_stokes4d_forcing_term(tcm, xcm, ycm, zcm, nu)

    # initial points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    zi = zc
    tim, xim, yim, zim = jnp.meshgrid(
        ti.ravel(), xi.ravel(), yi.ravel(), zi.ravel(), indexing='ij'
    )
    wi = navier_stokes4d_exact_w(tim, xim, yim, zim, nu)
    ui = navier_stokes4d_exact_u(tim, xim, yim, zim, nu)
    # boundary points (hard-coded)
    tb = [tc, tc, tc, tc, tc, tc]
    xb = [jnp.array([[-1.]]), jnp.array([[1.]]), xc, xc, xc, xc]
    yb = [yc, yc, jnp.array([[-1.]]), jnp.array([[1.]]), yc, yc]
    zb = [zc, zc, zc, zc, jnp.array([[-1.]]), jnp.array([[1.]])]
    wb = []
    for i in range(6):
        tbm, xbm, ybm, zbm = jnp.meshgrid(
            tb[i].ravel(), xb[i].ravel(), yb[i].ravel(), zb[i].ravel(), indexing='ij'
        )
        wb += [navier_stokes4d_exact_w(tbm, xbm, ybm, zbm, nu)]
    return tc, xc, yc, zc, fc, ti, xi, yi, zi, wi, ui, tb, xb, yb, zb, wb


#======================== Flow-Mixing 3-d ========================#
#----------------------------- PINN ------------------------------#
@partial(jax.jit, static_argnums=(0,))
def _pinn_train_generator_flow_mixing3d(nc, v_max, key):
    ni, nb = nc**2, nc**2

    keys = jax.random.split(key, 13)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc**3, 1), minval=0., maxval=4.)
    xc = jax.random.uniform(keys[1], (nc**3, 1), minval=-4., maxval=4.)
    yc = jax.random.uniform(keys[2], (nc**3, 1), minval=-4., maxval=4.)
    _, a, b = flow_mixing3d_params(tc, xc, yc, v_max, require_ab=True)

    # initial points
    ti = jnp.zeros((ni, 1))
    xi = jax.random.uniform(keys[3], (ni, 1), minval=-4., maxval=4.)
    yi = jax.random.uniform(keys[4], (ni, 1), minval=-4., maxval=4.)
    omega_i, _, _ = flow_mixing3d_params(ti, xi, yi, v_max)
    ui = flow_mixing3d_exact_u(ti, xi, yi, omega_i)

    # boundary points (hard-coded)
    tb = [
        jax.random.uniform(keys[5], (nb, 1), minval=0., maxval=4.),
        jax.random.uniform(keys[6], (nb, 1), minval=0., maxval=4.),
        jax.random.uniform(keys[7], (nb, 1), minval=0., maxval=4.),
        jax.random.uniform(keys[8], (nb, 1), minval=0., maxval=4.)
    ]
    xb = [
        jnp.array([[-4.]]*nb),
        jnp.array([[4.]]*nb),
        jax.random.uniform(keys[9], (nb, 1), minval=-4., maxval=4.),
        jax.random.uniform(keys[10], (nb, 1), minval=-4., maxval=4.)
    ]
    yb = [
        jax.random.uniform(keys[11], (nb, 1), minval=-4., maxval=4.),
        jax.random.uniform(keys[12], (nb, 1), minval=-4., maxval=4.),
        jnp.array([[-4.]]*nb),
        jnp.array([[4.]]*nb)
    ]
    ub = []
    for i in range(4):
        omega_b, _, _ = flow_mixing3d_params(tb[i], xb[i], yb[i], v_max)
        ub += [flow_mixing3d_exact_u(tb[i], xb[i], yb[i], omega_b)]
    tb = jnp.concatenate(tb)
    xb = jnp.concatenate(xb)
    yb = jnp.concatenate(yb)
    ub = jnp.concatenate(ub)
    return tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b


#----------------------------- SPINN, SPIKAN -----------------------------#
@partial(jax.jit, static_argnums=(0,))
def _spinn_train_generator_flow_mixing3d(nc, v_max, key):
    keys = jax.random.split(key, 3)
    # collocation points
    tc = jax.random.uniform(keys[0], (nc, 1), minval=0., maxval=4.)
    xc = jax.random.uniform(keys[1], (nc, 1), minval=-4., maxval=4.)
    yc = jax.random.uniform(keys[2], (nc, 1), minval=-4., maxval=4.)
    tc_mesh, xc_mesh, yc_mesh = jnp.meshgrid(tc.ravel(), xc.ravel(), yc.ravel(), indexing='ij')

    _, a, b = flow_mixing3d_params(tc_mesh, xc_mesh, yc_mesh, v_max, require_ab=True)

    # initial points
    ti = jnp.zeros((1, 1))
    xi = xc
    yi = yc
    ti_mesh, xi_mesh, yi_mesh = jnp.meshgrid(ti.ravel(), xi.ravel(), yi.ravel(), indexing='ij')
    omega_i, _, _ = flow_mixing3d_params(ti_mesh, xi_mesh, yi_mesh, v_max)
    ui = flow_mixing3d_exact_u(ti_mesh, xi_mesh, yi_mesh, omega_i)
    # boundary points (hard-coded)
    tb = [tc, tc, tc, tc]
    xb = [jnp.array([[-4.]]), jnp.array([[4.]]), xc, xc]
    yb = [yc, yc, jnp.array([[-4.]]), jnp.array([[4.]])]
    ub = []
    for i in range(4):
        tb_mesh, xb_mesh, yb_mesh = jnp.meshgrid(tb[i].ravel(), xb[i].ravel(), yb[i].ravel(), indexing='ij')
        omega_b, _, _ = flow_mixing3d_params(tb_mesh, xb_mesh, yb_mesh, v_max)
        ub += [flow_mixing3d_exact_u(tb_mesh, xb_mesh, yb_mesh, omega_b)]
    return tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b


#======================== Taylor-Couette flow2d ========================#
#---------------------------------- SPINN, SPIKAN -----------------------------------#
def _train_generator_taylor_couette_2d(args, key):
    R1, R2, Omega1, Omega2 = args.r1, args.r2, args.omega1, args.omega2
    nr_c, ntheta_c, n_b = args.nr_c, args.ntheta_c, args.n_b

    keys = jax.random.split(key, 2)

    #Collocation points
    rc = jax.random.uniform(keys[0], (nr_c, 1), minval=R1, maxval=R2)
    thetac = jax.random.uniform(keys[1], (ntheta_c, 1), minval=0., maxval=2.*jnp.pi)

    #Boundary points
    thetab = jnp.linspace(0, 2 * jnp.pi, n_b).reshape(-1, 1)
    
    # r = R1
    rb1 = jnp.full_like(thetab, R1)
    u_thetab1 = jnp.full_like(thetab, Omega1 * R1)
    
    # r = R2
    rb2 = jnp.full_like(thetab, R2)
    u_thetab2 = jnp.full_like(thetab, Omega2 * R2)
    
    rb = jnp.concatenate([rb1, rb2], axis=0)
    thetab_combined = jnp.concatenate([thetab, thetab], axis=0)
    u_thetab = jnp.concatenate([u_thetab1, u_thetab2], axis=0)

    return rc, thetac, rb, thetab_combined, u_thetab


def _train_generator_taylor_couette_2d_cartesian(args, key):
    R1, R2, Omega1, Omega2 = args.r1, args.r2, args.omega1, args.omega2
    n_c, n_b = args.n_c, args.n_b 
    
    keys = jax.random.split(key, 4)

    # Collocation points
    radius_c = jnp.sqrt(jax.random.uniform(keys[0], (n_c, 1), minval=R1**2, maxval=R2**2))
    angle_c = jax.random.uniform(keys[1], (n_c, 1), minval=0., maxval=2.*jnp.pi)
    xc = radius_c * jnp.cos(angle_c)
    yc = radius_c * jnp.sin(angle_c)

    # Boundary points
    angle_b = jax.random.uniform(keys[2], (n_b, 1), minval=0., maxval=2.*jnp.pi)
    # Inner boundary
    xb1 = R1 * jnp.cos(angle_b)
    yb1 = R1 * jnp.sin(angle_b)
    u_xb1 = -Omega1 * yb1
    u_yb1 = Omega1 * xb1
    # Outer boundary
    xb2 = R2 * jnp.cos(angle_b)
    yb2 = R2 * jnp.sin(angle_b)
    u_xb2 = -Omega2 * yb2
    u_yb2 = Omega2 * xb2
    
    xb = jnp.concatenate([xb1, xb2])
    yb = jnp.concatenate([yb1, yb2])
    u_xb = jnp.concatenate([u_xb1, u_xb2])
    u_yb = jnp.concatenate([u_yb1, u_yb2])

    return xc, yc, xb, yb, u_xb, u_yb



def generate_train_data(args, key, result_dir=None):
    eqn = args.equation
    if args.model == 'pinn':
        if eqn == 'flow_mixing3d':
            data = _pinn_train_generator_flow_mixing3d(
                args.nc, args.vmax, key
            )
        else:
            raise NotImplementedError
    elif args.model in ['spinn', 'spikan']:
        if eqn == 'navier_stokes3d':
            data = _spinn_train_generator_navier_stokes3d(
                args.nt, args.nxy, args.data_dir, result_dir, args.marching_steps, args.step_idx, args.offset_num, key
            )
        elif eqn == 'navier_stokes4d':
            data = _spinn_train_generator_navier_stokes4d(
                args.nc, args.nu, key
            )
        elif eqn == 'flow_mixing3d':
            data = _spinn_train_generator_flow_mixing3d(
                args.nc, args.vmax, key
            )
        elif eqn == 'taylor_couette_2d':
            data = _train_generator_taylor_couette_2d(
                args, key
            )
        elif args.equation == 'taylor_couette_2d_cartesian':
            data = _train_generator_taylor_couette_2d_cartesian(args, key)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return data


#============================== test dataset ===============================#
#----------------------- Navier-Stokes equation 3-d -------------------------#
def _test_generator_navier_stokes3d(model, data_dir, result_dir, marching_steps, step_idx):
    ns_data = scipy.io.loadmat(os.path.join(data_dir, 'w_data.mat'))
    t = ns_data['t'].reshape(-1, 1)
    x = ns_data['x'].reshape(-1, 1)
    y = ns_data['y'].reshape(-1, 1)
    t = jnp.insert(t, 0, jnp.array([0.]), axis=0)
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)

    gt = ns_data['w']   # without t=0
    gt = jnp.insert(gt, 0, ns_data['w0'], axis=0)

    # get data within current time window
    if marching_steps != 0:
        Dt = t[-1][0] / marching_steps  # interval of time window
        i = 0
        while Dt*(step_idx+1) > t[i][0]:
            i+=1
        t = t[:i]
        gt = gt[:i]

    # get data within current time window
    if step_idx > 0:
        w0_pred = scipy.io.loadmat(os.path.join(result_dir, '..', f'IC_pred/w0_{step_idx}.mat'))
        i = 0
        while t[i] != w0_pred['t'][0][0]:
            i+=1
        t = t[i:]
        gt = gt[i:]

    if model == 'pinn':
        tm, xm, ym = jnp.meshgrid(t.ravel(), x.ravel(), y.ravel(), indexing='ij')
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        gt = gt.reshape(-1, 1)
    
    return t, x, y, gt


#----------------------- Navier-Stokes equation 4-d -------------------------#
@partial(jax.jit, static_argnums=(0, 1,))
def _test_generator_navier_stokes4d(model, nc_test, nu):
    t = jnp.linspace(0, 5, nc_test)
    x = jnp.linspace(0, 2*jnp.pi, nc_test)
    y = jnp.linspace(0, 2*jnp.pi, nc_test)
    z = jnp.linspace(0, 2*jnp.pi, nc_test)
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    z = jax.lax.stop_gradient(z)
    tm, xm, ym, zm = jnp.meshgrid(
        t, x, y, z, indexing='ij'
    )
    w_gt = navier_stokes4d_exact_w(tm, xm, ym, zm, nu)
    if model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        z = zm.reshape(-1, 1)
        w_gt = w_gt.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
    return t, x, y, z, w_gt


#----------------------- Flow-Mixing 3-d -------------------------#
@partial(jax.jit, static_argnums=(0, 1,))
def _test_generator_flow_mixing3d(model, nc_test, v_max):
    t = jnp.linspace(0, 4, nc_test)
    x = jnp.linspace(-4, 4, nc_test)
    y = jnp.linspace(-4, 4, nc_test)
    t = jax.lax.stop_gradient(t)
    x = jax.lax.stop_gradient(x)
    y = jax.lax.stop_gradient(y)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')

    omega, _, _ = flow_mixing3d_params(tm, xm, ym, v_max)
    u_gt = flow_mixing3d_exact_u(tm, xm, ym, omega)

    if model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        u_gt = u_gt.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
    return t, x, y, u_gt


#======================== Taylor-Couette flow2d ========================#
def _test_generator_taylor_couette_2d(args):
    R1, R2, Omega1, Omega2 = args.r1, args.r2, args.omega1, args.omega2
    nr_eval, ntheta_eval = args.nr_eval, args.ntheta_eval

    A = (Omega2 * R2**2 - Omega1 * R1**2) / (R2**2 - R1**2)
    B = (Omega1 - Omega2) * R1**2 * R2**2 / (R2**2 - R1**2)

    r_vec = jnp.linspace(R1, R2, nr_eval).reshape(-1, 1)
    theta_vec = jnp.linspace(0, 2 * jnp.pi, ntheta_eval).reshape(-1, 1)
    r_grid, theta_grid = jnp.meshgrid(r_vec.ravel(), theta_vec.ravel(), indexing='ij')

    u_theta_gt = A * r_grid + B / r_grid
    
    return r_vec, theta_vec, u_theta_gt


def _test_generator_taylor_couette_2d_cartesian(args):
    nr_eval, ntheta_eval = args.nr_eval, args.ntheta_eval
    
    r_vec = jnp.linspace(args.r1, args.r2, nr_eval)
    theta_vec = jnp.linspace(0, 2 * jnp.pi, ntheta_eval)
    r_grid, theta_grid = jnp.meshgrid(r_vec, theta_vec, indexing='ij')
    
    x_grid = r_grid * jnp.cos(theta_grid)
    y_grid = r_grid * jnp.sin(theta_grid)
    
    u_theta_gt = taylor_couette_2d_exact_u(r_grid, args.r1, args.r2, args.omega1, args.omega2)
    u_x_gt = -u_theta_gt * jnp.sin(theta_grid)
    u_y_gt = u_theta_gt * jnp.cos(theta_grid)
    
    x_vec = jnp.linspace(-args.r2, args.r2, nr_eval).reshape(-1, 1)
    y_vec = jnp.linspace(-args.r2, args.r2, nr_eval).reshape(-1, 1)
    
    return x_vec, y_vec, x_grid, y_grid, u_x_gt, u_y_gt


def generate_test_data(args, result_dir):
    eqn = args.equation
    if eqn == 'navier_stokes3d':
        data = _test_generator_navier_stokes3d(
            args.model, args.data_dir, result_dir, args.marching_steps, args.step_idx
        )
    elif eqn == 'navier_stokes4d':
        data = _test_generator_navier_stokes4d(
            args.model, args.nc_test, args.nu
        )
    elif eqn == 'flow_mixing3d':
        data = _test_generator_flow_mixing3d(
            args.model, args.nc_test, args.vmax
        )
    elif eqn == 'taylor_couette_2d':
        data = _test_generator_taylor_couette_2d(args)
    elif args.equation == 'taylor_couette_2d_cartesian':
        data = _test_generator_taylor_couette_2d_cartesian(args)
    else:
        raise NotImplementedError
    return data