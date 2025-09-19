import jax
import jax.numpy as jnp
from functools import partial
from utils.vorticity import velocity_to_vorticity_fwd, velocity_to_vorticity_rev, vorx, vory, vorz
import pdb


def relative_l2(u, u_gt):
    return jnp.linalg.norm(u-u_gt) / jnp.linalg.norm(u_gt)

def mse(u, u_gt):
    return jnp.mean((u-u_gt)**2)

@partial(jax.jit, static_argnums=(0,))
def _eval2d(apply_fn, params, *test_data):
    x, y, u_gt = test_data
    return relative_l2(apply_fn(params, x, y), u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval2d_separable(apply_fn, params, *test_data):
    coord1, coord2, u_gt = test_data
    pred = apply_fn(params, coord1, coord2)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval3d(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = apply_fn(params, x, y, z)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval3d_ns_pinn(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = velocity_to_vorticity_rev(apply_fn, params, x, y, z)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval3d_ns_spinn(apply_fn, params, *test_data):
    x, y, z, u_gt = test_data
    pred = velocity_to_vorticity_fwd(apply_fn, params, x, y, z)
    return relative_l2(pred, u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval4d(apply_fn, params, *test_data):
    t, x, y, z, u_gt = test_data
    return relative_l2(apply_fn(params, t, x, y, z), u_gt)

@partial(jax.jit, static_argnums=(0,))
def _eval_ns4d(apply_fn, params, *test_data):
    t, x, y, z, w_gt = test_data
    error = 0
    wx = vorx(apply_fn, params, t, x, y, z)
    wy = vory(apply_fn, params, t, x, y, z)
    wz = vorz(apply_fn, params, t, x, y, z)
    error = relative_l2(wx, w_gt[0]) + relative_l2(wy, w_gt[1]) + relative_l2(wz, w_gt[2])
    return error / 3


# temporary code
def _batch_eval4d(apply_fn, params, *test_data):
    t, x, y, z, u_gt = test_data
    error, batch_size = 0., 100000
    n_iters = len(u_gt) // batch_size
    for i in range(n_iters):
        begin, end = i*batch_size, (i+1)*batch_size
        u = apply_fn(params, t[begin:end], x[begin:end], y[begin:end], z[begin:end])
        error += jnp.sum((u - u_gt[begin:end])**2)
    error = jnp.sqrt(error) / jnp.linalg.norm(u_gt)
    return error

@partial(jax.jit, static_argnums=(0,))
def _evalnd(apply_fn, params, *test_data):
    t, x_list, u_gt = test_data
    return relative_l2(apply_fn(params, t, *x_list), u_gt)


@partial(jax.jit, static_argnums=(0,))
def _eval2d_cartesian_velocity(apply_fn, params, *test_data, r1, r2):
    x_vec, y_vec, x_grid, y_grid, u_x_gt, u_y_gt = test_data
    
    u_x_pred, u_y_pred = apply_fn(params, x_vec, y_vec)
    
    pred_mag = jnp.sqrt(u_x_pred**2 + u_y_pred**2)
    gt_mag = jnp.sqrt(u_x_gt**2 + u_y_gt**2)

    r_pred = jnp.sqrt(x_grid**2 + y_grid**2)
    mask = (r_pred >= r1) & (r_pred <= r2)
    
    return relative_l2(pred_mag * mask, gt_mag * mask)


def setup_eval_function(model, equation, args = None):
    if 'taylor_couette_2d' in equation:
        dim = '2d'  
    else:
        dim = equation[-2:]

    if dim == '2d':
        if equation == 'taylor_couette_2d':
            fn = _eval2d_separable
        elif equation == 'taylor_couette_2d_cartesian':
            if args is None:
                raise ValueError("args must be provided for 'taylor_couette_2d_cartesian' evaluation.")
            fn = partial(_eval2d_cartesian_velocity, r1=args.r1, r2=args.r2)
        else:
            fn = _eval2d
    elif dim == '3d':
        if model == 'pinn' and equation == 'navier_stokes3d':
            fn = _eval3d_ns_pinn
        elif model in ['spinn', 'spikan'] and equation == 'navier_stokes3d':
            fn = _eval3d_ns_spinn
        else:
            fn = _eval3d
    elif dim == '4d':
        if model == 'pinn':
            fn = _batch_eval4d
        if model == 'spinn' and equation == 'navier_stokes4d':
            fn = _eval_ns4d
        else:
            fn = _eval4d
    elif dim == 'nd':
        if model == 'spinn':
            fn = _evalnd
    else:
        raise NotImplementedError
    return fn
