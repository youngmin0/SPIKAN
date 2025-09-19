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

# 유틸리티 함수들을 import 합니다.
from utils.data_generators import generate_train_data, generate_test_data
from utils.eval_functions import setup_eval_function
from utils.training_utils import setup_networks, name_model, save_config, update_model
from utils.visualizer import show_solution

# ==============================================================================
# 테일러-쿠에트 유동을 위한 손실 함수 (직교좌표계)
# ==============================================================================
@partial(jax.jit, static_argnums=(0,))
def loss_fn_taylor_couette_cartesian(apply_fn, params, xc, yc, xb, yb, u_xb, u_yb, lbda_b, nu):
    """
    직교좌표계에서의 테일러-쿠에트 유동에 대한 손실 함수.
    """
    
    # 1. Residual Loss (PDE Loss)
    def residual_loss(params, x, y):
        # 모델은 [u_x, u_y]를 리스트 형태로 예측합니다.
        u, v = apply_fn(params, x, y)
        
        # JAX의 자동 미분을 사용하여 각 미분 항을 계산합니다.
        vec_x = jnp.ones(x.shape)
        vec_y = jnp.ones(y.shape)

        u_x = jvp(lambda x: apply_fn(params, x, y)[0], (x,), (vec_x,))[1]
        u_y = jvp(lambda y: apply_fn(params, x, y)[0], (y,), (vec_y,))[1]
        v_x = jvp(lambda x: apply_fn(params, x, y)[1], (x,), (vec_x,))[1]
        v_y = jvp(lambda y: apply_fn(params, x, y)[1], (y,), (vec_y,))[1]

        u_xx = hvp_fwdfwd(lambda x: apply_fn(params, x, y)[0], (x,), (vec_x,))
        u_yy = hvp_fwdfwd(lambda y: apply_fn(params, x, y)[0], (y,), (vec_y,))
        v_xx = hvp_fwdfwd(lambda x: apply_fn(params, x, y)[1], (x,), (vec_x,))
        v_yy = hvp_fwdfwd(lambda y: apply_fn(params, x, y)[1], (y,), (vec_y,))

        # 정상 상태 비압축성 Navier-Stokes 방정식 잔차
        # x-momentum
        res_x = u * u_x + v * u_y - nu * (u_xx + u_yy)
        # y-momentum
        res_y = u * v_x + v * v_y - nu * (v_xx + v_yy)
        # Incompressibility (Continuity)
        res_c = u_x + v_y

        return jnp.mean(res_x**2) + jnp.mean(res_y**2) + jnp.mean(res_c**2)

    # 2. Boundary Loss
    def boundary_loss(params, x_b, y_b, u_x_b, u_y_b):
        u_pred_b, v_pred_b = apply_fn(params, x_b, y_b)
        loss_b = jnp.mean((u_pred_b - u_x_b)**2) + jnp.mean((v_pred_b - u_y_b)**2)
        return loss_b

    # 총 손실 계산
    loss_r = residual_loss(params, xc, yc)
    loss_b = boundary_loss(params, xb, yb, u_xb, u_yb)
    total_loss = loss_r + lbda_b * loss_b
    
    gradient = jax.grad(lambda p: residual_loss(p, xc, yc) + lbda_b * boundary_loss(p, xb, yb, u_xb, u_yb))(params)
    return total_loss, gradient


# ==============================================================================
# 메인 실행 블록
# ==============================================================================
if __name__ == '__main__':
    # --- 1. Argument Parser 설정 ---
    parser = argparse.ArgumentParser(description='Taylor-Couette Flow in Cartesian Coordinates')

    # --- 실험 기본 설정 ---
    parser.add_argument('--equation', type=str, default='taylor_couette_2d_cartesian', help='Equation to solve')
    parser.add_argument('--model', type=str, default='spikan', choices=['spinn', 'spikan'], help='Model name')
    parser.add_argument('--seed', type=int, default=111, help='Random seed')
    parser.add_argument('--epochs', type=int, default=50000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')

    # --- 물리적 파라미터 ---
    parser.add_argument('--nu', type=float, default=0.1, help='Kinematic viscosity')
    parser.add_argument('--r1', type=float, default=1.0, help='Inner cylinder radius')
    parser.add_argument('--r2', type=float, default=2.0, help='Outer cylinder radius')
    parser.add_argument('--omega1', type=float, default=1.0, help='Inner cylinder angular velocity')
    parser.add_argument('--omega2', type=float, default=0.0, help='Outer cylinder angular velocity')

    # --- 데이터 샘플링 파라미터 ---
    parser.add_argument('--n_c', type=int, default=10000, help='Number of collocation points')
    parser.add_argument('--n_b', type=int, default=1000, help='Number of boundary points on each cylinder')
    parser.add_argument('--nr_eval', type=int, default=100, help='Grid resolution for evaluation (r-axis)')
    parser.add_argument('--ntheta_eval', type=int, default=200, help='Grid resolution for evaluation (theta-axis)')

    # --- 손실 가중치 ---
    parser.add_argument('--lbda_b', type=float, default=100.0, help='Weighting factor for boundary condition')

    # --- 모델 구조 파라미터 ---
    parser.add_argument('--mlp', type=str, default='modified_mlp', help='MLP type for SPINN')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--features', type=int, default=20, help='Feature size of each layer')
    parser.add_argument('--r', type=int, default=20, help='Rank of the approximated tensor')
    
    # --- KAN 전용 파라미터 ---
    kan_parser = parser.add_argument_group('KAN settings for SPIKAN model')
    kan_parser.add_argument('--kan_k', type=int, default=3, help='Order of B-spline for KAN layer')
    kan_parser.add_argument('--kan_g', type=int, default=10, help='Number of grid intervals for KAN layer')
    
    # --- 로그 설정 ---
    parser.add_argument('--log_iter', type=int, default=1000, help='Print log every...')
    parser.add_argument('--plot_iter', type=int, default=10000, help='Plot result every...')
    
    args = parser.parse_args()
    # 직교좌표계 모델은 u_x, u_y 두 개를 예측해야 하므로 out_dim=2로 고정
    args.out_dim = 2

    # --- 2. 초기 설정 ---
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

    # --- 3. 데이터 생성 ---
    key, subkey = jax.random.split(key, 2)
    train_data = generate_train_data(args, subkey)
    test_data = generate_test_data(args, result_dir)
    xc, yc, xb, yb, u_xb, u_yb = train_data

    eval_fn = setup_eval_function(args.model, args.equation, args)
    
    log_file_path = os.path.join(result_dir, 'log (loss, error).csv')
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # --- 4. 학습 루프 ---
    print("="*50)
    print(f"Starting training for {name}")
    print(f"Total parameters: {args.total_params}")
    print("="*50)
    
    start_time = time.time()
    for e in trange(1, args.epochs + 1):
        loss, gradient = loss_fn_taylor_couette_cartesian(apply_fn, params, xc, yc, xb, yb, u_xb, u_yb, args.lbda_b, args.nu)
        params, state = update_model(optim, gradient, params, state)

        if e % args.log_iter == 0:
            error = eval_fn(apply_fn, params, *test_data)
            print(f'Epoch: {e}/{args.epochs} --> Total Loss: {loss:.6f}, L2 Error: {error:.6f}')
            with open(log_file_path, 'a') as f:
                f.write(f'{loss},{error}\n')

        if e % args.plot_iter == 0:
            show_solution(args, apply_fn, params, test_data, result_dir, e)
    
    # --- 5. 학습 완료 ---
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Training finished in {runtime:.2f} seconds.")
    
    jnp.save(os.path.join(result_dir, 'params.npy'), params)
    np.savetxt(os.path.join(result_dir, 'total_runtime (sec).csv'), np.array([runtime]), delimiter=',')
    
    show_solution(args, apply_fn, params, test_data, result_dir, args.epochs)
    print(f"Results saved in: {result_dir}")
