import os
import pdb
from functools import partial

import jax
import jax.numpy as jnp
import optax
import scipy.io
from networks.physics_informed_neural_networks import *
from utils.vorticity import (velocity_to_vorticity_fwd,
                             velocity_to_vorticity_rev)


def setup_networks(args, key):
    # build network
    if 'taylor_couette_2d' in args.equation:
        dim = '2d'
    else:
        dim = args.equation[-2:]
    
    if args.model == 'pinn':
        # feature sizes
        feat_sizes = tuple([args.features for _ in range(args.n_layers - 1)] + [args.out_dim])
        if dim == '2d':
            model = PINN2d(feat_sizes)
        elif dim == '3d':
            model = PINN3d(feat_sizes, args.out_dim, args.pos_enc)
        elif dim == '4d':
            model = PINN4d(feat_sizes)
        else:
            raise NotImplementedError
    else:  # SPINN or SPIKAN
        feat_sizes = tuple([args.features for _ in range(args.n_layers)])

        if dim == '2d':
            if args.model == 'spinn':
                model = SPINN2d(feat_sizes, args.r, args.out_dim, args.mlp)
            elif args.model == 'spikan':
                model = SPIKAN2d(
                    features=feat_sizes, r=args.r, out_dim=args.out_dim,
                    kan_k=args.kan_k, kan_g=args.kan_g
                )
        elif dim == '3d':
            if args.model == 'spinn':
                model = SPINN3d(feat_sizes, args.r, args.out_dim, args.pos_enc, args.mlp)
            elif args.model == 'spikan':
                model = SPIKAN3d(
                    features=feat_sizes, r=args.r, out_dim=args.out_dim,
                    pos_enc=args.pos_enc, kan_k=args.kan_k, kan_g=args.kan_g
                )
        elif dim == '4d':
            if args.model == 'spinn':
                model = SPINN4d(feat_sizes, args.r, args.out_dim, args.mlp)
            else:
                 raise NotImplementedError(f"SPIKAN not implemented for {dim}")
        else:
            raise NotImplementedError(f"Model not implemented for {dim}")
    # initialize params
    # dummy inputs must be given
    if dim == '2d':
        if args.equation == 'taylor_couette_2d':
            params = model.init(
                key,
                jnp.ones((args.nr_c, 1)),
                jnp.ones((args.ntheta_c, 1))
            )
        else:
             params = model.init(
                key,
                jnp.ones((args.n_c, 1)),
                jnp.ones((args.n_c, 1))
            )
    elif dim == '3d':
        if args.equation == 'navier_stokes3d':
            params = model.init(
                key,
                jnp.ones((args.nt, 1)),
                jnp.ones((args.nxy, 1)),
                jnp.ones((args.nxy, 1))
            )
        else:
            params = model.init(
                key,
                jnp.ones((args.nc, 1)),
                jnp.ones((args.nc, 1)),
                jnp.ones((args.nc, 1))
            )
    elif dim == '4d':
        params = model.init(
            key,
            jnp.ones((args.nc, 1)),
            jnp.ones((args.nc, 1)),
            jnp.ones((args.nc, 1)),
            jnp.ones((args.nc, 1))
        )
    else:
        raise NotImplementedError

    return jax.jit(model.apply), params


def name_model(args):
    name = [
        f'nl{args.n_layers}', f'fs{args.features}',
        f'lr{args.lr}', f's{args.seed}',
    ]

    if args.model in ['spinn', 'spikan']:
        name.append(f'r{args.r}')

    if args.equation == 'navier_stokes3d':
        name.insert(0, f'nxy{args.nxy}')
        name.insert(0, f'nt{args.nt}')
        name.append(f'on{args.offset_num}')
        name.append(f'oi{args.offset_iter}')
        name.append(f'lc{args.lbda_c}')
        name.append(f'lic{args.lbda_ic}')
    
    elif args.equation == 'taylor_couette_2d':
        name.insert(0, f'ntheta_c{args.ntheta_c}')
        name.insert(0, f'nr_c{args.nr_c}')
        name.append(f'r1_{args.r1}')
        name.append(f'r2_{args.r2}')
        name.append(f'o1_{args.omega1}')
        name.append(f'o2_{args.omega2}')
        name.append(f'lbda_b{args.lbda_b}')

    elif args.equation == 'taylor_couette_2d_cartesian':
        name.insert(0, f'nb{args.n_b}')
        name.insert(0, f'nc{args.n_c}')
        name.append(f'r1_{args.r1}')
        name.append(f'r2_{args.r2}')
        name.append(f'o1_{args.omega1}')
        name.append(f'o2_{args.omega2}')
        name.append(f'lbda_b{args.lbda_b}')

    if args.model == 'spinn':
        name.append(f'{args.mlp}')
    elif args.model == 'spikan':
        name.append(f'k{args.kan_k}')
        name.append(f'g{args.kan_g}')
        
    return '_'.join(name)


def save_config(args, result_dir):
    with open(os.path.join(result_dir, 'configs.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')


# single update function
@partial(jax.jit, static_argnums=(0,))
def update_model(optim, gradient, params, state):
    updates, state = optim.update(gradient, state)
    params = optax.apply_updates(params, updates)
    return params, state


# save next initial condition for time-marching
def save_next_IC(root_dir, name, apply_fn, params, test_data, step_idx, e):
    os.makedirs(os.path.join(root_dir, name, 'IC_pred'), exist_ok=True)

    w_pred = velocity_to_vorticity_fwd(apply_fn, params, jnp.expand_dims(test_data[0][-1], axis=1), test_data[1], test_data[2])
    w_pred = w_pred.reshape(-1, test_data[1].shape[0], test_data[2].shape[0])[0]
    u0_pred, v0_pred = apply_fn(params, jnp.expand_dims(test_data[0][-1], axis=1), test_data[1], test_data[2])
    u0_pred, v0_pred = jnp.squeeze(u0_pred), jnp.squeeze(v0_pred)
    
    scipy.io.savemat(os.path.join(root_dir, name, f'IC_pred/w0_{step_idx+1}.mat'), mdict={'w0': w_pred, 'u0': u0_pred, 'v0': v0_pred, 't': jnp.expand_dims(test_data[0][-1], axis=1)})