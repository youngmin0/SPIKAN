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
def loss_fn_taylor_couette_cartesian(apply_fn, params, xc, yc, xb, yb, u_xb, u_yb, lbda_b, nu):
    # Residual Loss (PDE Loss)
    def residual_loss(params, x, y):
        outputs = apply_fn(params, x, y)
        
        # [수정] 모델 출력이 리스트인지 배열인지 확인하여 처리
        if isinstance(outputs, (list, tuple)):
            u = outputs[0]
            v = outputs[1]
            p = outputs[2]
        else:
            u = outputs[..., 0:1]
            v = outputs[..., 1:2]
            p = outputs[..., 2:3]

        vec_x = jnp.ones(x.shape)
        vec_y = jnp.ones(y.shape)

        # [수정] 미분을 위한 헬퍼 함수들도 리스트/배열 모두 대응하도록 수정
        def get_component(x, y, idx):
            out = apply_fn(params, x, y)
            if isinstance(out, (list, tuple)):
                return out[idx]
            return out[..., idx:idx+1]

        # 1. u 미분
        u_fn = lambda x, y: get_component(x, y, 0)
        u_x = jvp(lambda x: u_fn(x, y), (x,), (vec_x,))[1]
        u_y = jvp(lambda y: u_fn(x, y), (y,), (vec_y,))[1]
        u_xx = hvp_fwdfwd(lambda x: u_fn(x, y), (x,), (vec_x,))
        u_yy = hvp_fwdfwd(lambda y: u_fn(x, y), (y,), (vec_y,))

        # 2. v 미분
        v_fn = lambda x, y: get_component(x, y, 1)
        v_x = jvp(lambda x: v_fn(x, y), (x,), (vec_x,))[1]
        v_y = jvp(lambda y: v_fn(x, y), (y,), (vec_y,))[1]
        v_xx = hvp_fwdfwd(lambda x: v_fn(x, y), (x,), (vec_x,))
        v_yy = hvp_fwdfwd(lambda y: v_fn(x, y), (y,), (vec_y,))

        # 3. p 미분
        p_fn = lambda x, y: get_component(x, y, 2)
        p_x = jvp(lambda x: p_fn(x, y), (x,), (vec_x,))[1]
        p_y = jvp(lambda y: p_fn(x, y), (y,), (vec_y,))[1]

        # 4. Navier-Stokes Momentum Equations
        res_x = (u * u_x + v * u_y) + p_x - nu * (u_xx + u_yy)
        res_y = (u * v_x + v * v_y) + p_y - nu * (v_xx + v_yy)
        res_c = u_x + v_y

        return jnp.mean(res_x**2) + jnp.mean(res_y**2) + jnp.mean(res_c**2)

    # Boundary Loss
    def boundary_loss(params, x_b, y_b, u_x_b, u_y_b):
        outputs = apply_fn(params, x_b, y_b)
        
        # [수정] 경계 조건 처리도 리스트 대응
        if isinstance(outputs, (list, tuple)):
            u_pred_b = outputs[0]
            v_pred_b = outputs[1]
        else:
            u_pred_b = outputs[..., 0:1]
            v_pred_b = outputs[..., 1:2]
        
        loss_b = jnp.mean((u_pred_b - u_x_b)**2) + jnp.mean((v_pred_b - u_y_b)**2)
        return loss_b

    loss_r = residual_loss(params, xc, yc)
    loss_b = boundary_loss(params, xb, yb, u_xb, u_yb)
    
    total_loss = loss_r + lbda_b * loss_b
    
    gradient = jax.grad(lambda p: residual_loss(p, xc, yc) + lbda_b * boundary_loss(p, xb, yb, u_xb, u_yb))(params)
    return total_loss, gradient

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Taylor-Couette Flow in Cartesian Coordinates')

    parser.add_argument('--equation', type=str, default='taylor_couette_2d_cartesian', help='Equation to solve')
    parser.add_argument('--model', type=str, default='spikan', choices=['spinn', 'spikan'], help='Model name')
    parser.add_argument('--seed', type=int, default=111, help='Random seed')
    parser.add_argument('--epochs', type=int, default=50000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')

    parser.add_argument('--nu', type=float, default=0.1, help='Kinematic viscosity')
    parser.add_argument('--r1', type=float, default=1.0, help='Inner cylinder radius')
    parser.add_argument('--r2', type=float, default=2.0, help='Outer cylinder radius')
    parser.add_argument('--omega1', type=float, default=1.0, help='Inner cylinder angular velocity')
    parser.add_argument('--omega2', type=float, default=0.0, help='Outer cylinder angular velocity')

    parser.add_argument('--n_c', type=int, default=10000, help='Number of collocation points')
    parser.add_argument('--n_b', type=int, default=1000, help='Number of boundary points on each cylinder')
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
    
    # [중요 수정] u, v, p 세 가지 값을 예측해야 하므로 출력 차원을 3으로 설정
    args.out_dim = 3 

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key, 2)
    
    apply_fn, params = setup_networks(args, subkey)
    args.total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    
    name = name_model(args)
    root_dir = os.path.join(os.getcwd(), 'results', args.equation, args.model)
    result_dir = os.path.join(root_dir, name)
    os.makedirs(result_dir, exist_ok=True)
    
    optim = optax.adam(learning_rate=args.lr)
    state = optim.init(params)
    
    save_config(args, result_dir)

    key, subkey = jax.random.split(key, 2)
    train_data = generate_train_data(args, subkey)
    test_data = generate_test_data(args, result_dir)
    xc, yc, xb, yb, u_xb, u_yb = train_data

    eval_fn = setup_eval_function(args.model, args.equation, args)
    
    log_file_path = os.path.join(result_dir, 'log (loss, error).csv')
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    print("="*50)
    print(f"Starting training for {name}")
    print(f"Total parameters: {args.total_params}")
    print("="*50)
    
    start_time = time.time()
    for e in trange(1, args.epochs + 1):
        loss, gradient = loss_fn_taylor_couette_cartesian(apply_fn, params, xc, yc, xb, yb, u_xb, u_yb, args.lbda_b, args.nu)
        params, state = update_model(optim, gradient, params, state)

        if e % args.log_iter == 0:
            # eval_fn은 보통 u, v만 비교하도록 설정되어 있을 수 있습니다.
            # args.out_dim=3이어도 eval_fn 내부에서 앞의 2개만 쓰도록 되어있다면 문제없습니다.
            # 만약 차원 에러가 난다면 utils/eval_functions.py 확인이 필요합니다.
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