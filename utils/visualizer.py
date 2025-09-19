import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils.vorticity import velocity_to_vorticity_fwd
from utils.vorticity import vorx, vory, vorz

import pdb


def _navier_stokes3d(apply_fn, params, test_data, result_dir, e):
    print("visualizing solution...")

    nt, nx, ny = test_data[0].shape[0], test_data[1].shape[0], test_data[2].shape[0]

    t = test_data[0][-1]
    t = jnp.expand_dims(t, axis=1)

    w_pred = velocity_to_vorticity_fwd(apply_fn, params, t, test_data[1], test_data[2])
    w_pred = w_pred.reshape(-1, nx, ny)
    w_ref = test_data[-1][-1]

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)

    fig = plt.figure(figsize=(14, 5))

    # reference solution
    ax1 = fig.add_subplot(131)
    im = ax1.imshow(w_ref, cmap='jet', extent=[0, 2*jnp.pi, 0, 2*jnp.pi], vmin=jnp.min(w_ref), vmax=jnp.max(w_ref))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title(f'Reference $\omega(t={jnp.round(t[0][0], 1):.2f}, x, y)$', fontsize=15)

    # predicted solution
    ax1 = fig.add_subplot(132)
    im = ax1.imshow(w_pred[0], cmap='jet', extent=[0, 2*jnp.pi, 0, 2*jnp.pi], vmin=jnp.min(w_ref), vmax=jnp.max(w_ref))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title(f'Predicted $\omega(t={jnp.round(t[0][0], 1):.2f}, x, y)$', fontsize=15)

    # absolute error
    ax1 = fig.add_subplot(133)
    im = ax1.imshow(jnp.abs(w_ref - w_pred[0]), cmap='jet', extent=[0, 2*jnp.pi, 0, 2*jnp.pi], vmin=jnp.min(w_ref), vmax=jnp.max(w_ref))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_title(f'Asolute error', fontsize=15)

    cbar_ax = fig.add_axes([0.95, 0.3, 0.01, 0.4])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'))
    plt.close()


def _navier_stokes4d(apply_fn, params, test_data, result_dir, e):
    print("visualizing solution...")

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)

    fig = plt.figure(figsize=(30, 5))
    for t, sub in zip([0, 1, 2, 3, 4, 5], [161, 162, 163, 164, 165, 166]):
        t = jnp.array([[t]])
        x = jnp.linspace(0, 2*jnp.pi, 4).reshape(-1, 1)
        y = jnp.linspace(0, 2*jnp.pi, 30).reshape(-1, 1)
        z = jnp.linspace(0, 2*jnp.pi, 30).reshape(-1, 1)
        wx = vorx(apply_fn, params, t, x, y, z)
        wy = vory(apply_fn, params, t, x, y, z)
        wz = vorz(apply_fn, params, t, x, y, z)

        # c = jnp.sqrt(u_x**2 + u_y**2 + u_z**2)   # magnitude
        c = jnp.arctan2(wy, wz)    # zenith angle
        c = (c.ravel() - c.min()) / c.ptp()
        c = jnp.concatenate((c, jnp.repeat(c, 2)))
        c = plt.cm.plasma(c)

        x, y, z = jnp.meshgrid(jnp.squeeze(x), jnp.squeeze(y), jnp.squeeze(z), indexing='ij')

        ax = fig.add_subplot(sub, projection='3d')
        ax.quiver(x, y, z, jnp.squeeze(wx), jnp.squeeze(wy), jnp.squeeze(wz), length=0.1, colors=c, alpha=1, linewidth=0.7)
        plt.title(f't={jnp.squeeze(t)}')
    
    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'))
    plt.close()

def _taylor_couette_2d(apply_fn, params, test_data, result_dir, e):
    print("visualizing solution...")

    r_vec, theta_vec, u_gt = test_data

    u_pred = apply_fn(params, r_vec, theta_vec)

    r_grid, theta_grid = np.meshgrid(r_vec.ravel(), theta_vec.ravel(), indexing='ij')
    
    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw=dict(projection='polar'))
    fig.suptitle(f'Taylor-Couette Flow at Epoch {e}', fontsize=16)

    vmin = jnp.min(u_gt)
    vmax = jnp.max(u_gt)

    # 1. Reference solution
    ax1 = axes[0]
    c1 = ax1.pcolormesh(theta_grid, r_grid, u_gt, cmap='jet', vmin=vmin, vmax=vmax)
    ax1.set_title('Reference $u_\\theta(r, \\theta)$', fontsize=15, pad=20)
    ax1.set_yticklabels([]) 

    # Predicted solution 
    ax2 = axes[1]
    c2 = ax2.pcolormesh(theta_grid, r_grid, u_pred, cmap='jet', vmin=vmin, vmax=vmax)
    ax2.set_title('Predicted $u_\\theta(r, \\theta)$', fontsize=15, pad=20)
    ax2.set_yticklabels([])

    # Absolute error
    ax3 = axes[2]
    error = jnp.abs(u_gt - u_pred)
    c3 = ax3.pcolormesh(theta_grid, r_grid, error, cmap='inferno', vmin=0, vmax=jnp.max(error))
    ax3.set_title('Absolute Error', fontsize=15, pad=20)
    ax3.set_yticklabels([])

    fig.colorbar(c1, ax=axes[0:2], orientation='vertical', fraction=0.046, pad=0.04)
    fig.colorbar(c3, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'))
    plt.close()

def _taylor_couette_2d_cartesian(apply_fn, params, test_data, result_dir, e):
    print("visualizing solution...")
    x_vec, y_vec, x_grid, y_grid, u_x_gt, u_y_gt = test_data
    u_x_pred, u_y_pred = apply_fn(params, x_vec, y_vec)
    
    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f'Taylor-Couette Flow (Cartesian) at Epoch {e}', fontsize=16)

    gt_mag = jnp.sqrt(u_x_gt**2 + u_y_gt**2)
    pred_mag = jnp.sqrt(u_x_pred**2 + u_y_pred**2)
    error_mag = jnp.abs(gt_mag - pred_mag)

    vmin, vmax = jnp.min(gt_mag), jnp.max(gt_mag)
    skip = 8

    # 1. Reference solution
    axes[0].set_title('Reference Velocity Field', fontsize=15)
    axes[0].contourf(x_grid, y_grid, gt_mag, levels=50, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip], u_x_gt[::skip, ::skip], u_y_gt[::skip, ::skip], color='white')
    axes[0].set_aspect('equal', 'box')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    # 2. Predicted solution
    axes[1].set_title('Predicted Velocity Field', fontsize=15)
    axes[1].contourf(x_grid, y_grid, pred_mag, levels=50, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip], u_x_pred[::skip, ::skip], u_y_pred[::skip, ::skip], color='white')
    axes[1].set_aspect('equal', 'box')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    
    # 3. Absolute error
    axes[2].set_title('Absolute Error in Magnitude', fontsize=15)
    im = axes[2].contourf(x_grid, y_grid, error_mag, levels=50, cmap='inferno')
    axes[2].set_aspect('equal', 'box')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'))
    plt.close()

def show_solution(args, apply_fn, params, test_data, result_dir, e, resol=50):
    if args.equation == 'navier_stokes3d':
        _navier_stokes3d(apply_fn, params, test_data, result_dir, e)
    elif args.equation == 'navier_stokes4d':
        _navier_stokes4d(apply_fn, params, test_data, result_dir, e)
    elif args.equation == 'taylor_couette_2d':
        _taylor_couette_2d(apply_fn, params, test_data, result_dir, e)
    elif args.equation == 'taylor_couette_2d_cartesian':
        _taylor_couette_2d_cartesian(apply_fn, params, test_data, result_dir, e)
    else:
        raise NotImplementedError