import argparse
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import jvp
from functools import partial
from networks.hessian_vector_products import hvp_fwdfwd
from tqdm import trange

from utils.data_generators import generate_train_data, generate_test_data
from utils.eval_functions import setup_eval_function
from utils.training_utils import setup_networks, name_model, save_config, update_model
from utils.visualizer import show_solution


@partial(jax.jit, static_argnums=(0,))
def loss_fn_taylor_couette(apply_fn, params, rc, thetac, rb, thetab, u_thetab, lbda_b, nu):
    # 1. Residual Loss (PDE Loss)
    def residual_loss(params, r, theta):
        v_fn = lambda r: r * apply_fn(params, r, theta)
        
        vec_r = jnp.ones(r.shape)
        dv_dr = jvp(v_fn, (r,), (vec_r,))[1]
        d2v_dr2 = hvp_fwdfwd(v_fn, (r,), (vec_r,))

        residual = (1.0 / r) * d2v_dr2 - (1.0 / r**2) * dv_dr
        
        return jnp.mean(residual**2)

    # 2. Boundary Loss
    def boundary_loss(params, r_b, theta_b, u_theta_b):
        u_theta_pred_b = apply_fn(params, r_b, theta_b)
        loss_b = jnp.mean((u_theta_pred_b - u_theta_b)**2)
        return loss_b

    loss_r = residual_loss(params, rc, thetac)
    loss_b = boundary_loss(params, rb, thetab, u_thetab)
    
    total_loss = loss_r + lbda_b * loss_b
    
    gradient = jax.grad(lambda p: residual_loss(p, rc, thetac) + lbda_b * boundary_loss(p, rb, thetab, u_thetab))(params)

    return total_loss, gradient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Taylor-Couette Flow Training Configurations')

    parser.add_argument('--equation', type=str, default='taylor_couette_2d', help='Equation to solve')
    parser.add_argument('--model', type=str, default='spikan', choices=['spinn', 'spikan'], help='Model name')
    parser.add_argument('--seed', type=int, default=111, help='Random seed')
    parser.add_argument('--epochs', type=int, default=50000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')

    lr_decay_parser = parser.add_argument_group('Learning Rate Decay Settings')
    lr_decay_parser.add_argument('--lr_decay_steps', type=int, default=5000, help='Steps over which learning rate decays.')
    lr_decay_parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='Decay rate.')

    parser.add_argument('--nu', type=float, default=0.1, help='Kinematic viscosity')
    parser.add_argument('--r1', type=float, default=1.0, help='Inner cylinder radius')
    parser.add_argument('--r2', type=float, default=2.0, help='Outer cylinder radius')
    parser.add_argument('--omega1', type=float, default=1.0, help='Inner cylinder angular velocity')
    parser.add_argument('--omega2', type=float, default=0.0, help='Outer cylinder angular velocity')

    parser.add_argument('--nr_c', type=int, default=100, help='Number of collocation points for r-axis')
    parser.add_argument('--ntheta_c', type=int, default=200, help='Number of collocation points for theta-axis')
    parser.add_argument('--n_b', type=int, default=200, help='Number of boundary points on each cylinder')
    parser.add_argument('--nr_eval', type=int, default=100, help='Grid resolution for evaluation (r-axis)')
    parser.add_argument('--ntheta_eval', type=int, default=200, help='Grid resolution for evaluation (theta-axis)')

    parser.add_argument('--lbda_b', type=float, default=100.0, help='Weighting factor for boundary condition')

    parser.add_argument('--mlp', type=str, default='modified_mlp', help='MLP type for SPINN')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--features', type=int, default=20, help='Feature size of each layer')
    parser.add_argument('--r', type=int, default=20, help='Rank of the approximated tensor')

    kan_parser = parser.add_argument_group('KAN settings for SPIKAN model')
    kan_parser.add_argument('--kan_k', type=int, default=3, help='Order of B-spline for KAN layer')
    kan_parser.add_argument('--kan_g', type=int, default=10, help='Number of grid intervals for KAN layer')

    parser.add_argument('--log_iter', type=int, default=1000, help='Print log every...')
    parser.add_argument('--plot_iter', type=int, default=10000, help='Plot result every...')
    
    args = parser.parse_args()
    args.out_dim = 1

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key, 2)

    apply_fn, params = setup_networks(args, subkey)
    args.total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    name = name_model(args)
    root_dir = os.path.join(os.getcwd(), 'results', args.equation, args.model)
    result_dir = os.path.join(root_dir, name)
    os.makedirs(result_dir, exist_ok=True)

    lr_schedule = optax.exponential_decay(
        init_value=args.lr,
        transition_steps=args.lr_decay_steps,
        decay_rate=args.lr_decay_rate
    )

    optim = optax.adam(learning_rate=lr_schedule)
    state = optim.init(params)

    save_config(args, result_dir)

    key, subkey = jax.random.split(key, 2)
    train_data = generate_train_data(args, subkey)
    test_data = generate_test_data(args, result_dir)
    rc, thetac, rb, thetab, u_thetab = train_data

    eval_fn = setup_eval_function(args.model, args.equation)

    log_file_path = os.path.join(result_dir, 'log (loss, error).csv')
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    print("="*50)
    print(f"Starting training for {name}")
    print(f"Total parameters: {args.total_params}")
    print("="*50)
    
    start_time = time.time()
    for e in trange(1, args.epochs + 1):
        loss, gradient = loss_fn_taylor_couette(apply_fn, params, rc, thetac, rb, thetab, u_thetab, args.lbda_b, args.nu)

        params, state = update_model(optim, gradient, params, state)

        if e % args.log_iter == 0:
            error = eval_fn(apply_fn, params, *test_data)
            print(f'Epoch: {e}/{args.epochs} --> Total Loss: {loss:.6f}, L2 Error: {error:.6f}')
            with open(log_file_path, 'a') as f:
                f.write(f'{loss},{error}\n')

        if e % args.plot_iter == 0:
            show_solution(args, apply_fn, params, test_data, result_dir, e)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Training finished in {runtime:.2f} seconds.")
    
    jnp.save(os.path.join(result_dir, 'params.npy'), params)
    np.savetxt(os.path.join(result_dir, 'total_runtime (sec).csv'), np.array([runtime]), delimiter=',')
    
    show_solution(args, apply_fn, params, test_data, result_dir, args.epochs)
    print(f"Results saved in: {result_dir}")
