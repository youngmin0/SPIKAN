import os

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



def show_solution(args, apply_fn, params, test_data, result_dir, e, resol=50):
    if args.equation == 'navier_stokes3d':
        _navier_stokes3d(apply_fn, params, test_data, result_dir, e)
    elif args.equation == 'navier_stokes4d':
        _navier_stokes4d(apply_fn, params, test_data, result_dir, e)
    else:
        raise NotImplementedError