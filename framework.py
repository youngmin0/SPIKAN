import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import jax
import jax.numpy as jnp
from networks.physics_informed_neural_networks import PINN2d

def calculate_si(matrix):
    try:
        matrix_np = np.array(matrix)
        singular_values = svd(matrix_np, compute_uv=False)
        if np.sum(singular_values) < 1e-9: return 0.0, singular_values
        si = singular_values[0] / np.sum(singular_values)
        return si, singular_values
    except Exception as e:
        print(f"SVD 계산 중 오류 발생: {e}")
        return 0.0, []

def estimate_reff(singular_values, threshold=0.999):
    if np.sum(singular_values) < 1e-9: return 0, np.array([0])
    normalized_s = singular_values / np.sum(singular_values)
    cumulative_energy = np.cumsum(normalized_s)
    reff = np.searchsorted(cumulative_energy, threshold, side='right') + 1
    return reff, cumulative_energy


# --- 저비용 JAX PINN으로 저해상도 해 생성 ---
grid_res = 50
R1, R2 = 1.0, 2.0

def get_low_res_solution_jax(is_polar, key):
    low_cost_features = [16, 16, 2]
    model = PINN2d(features=low_cost_features)

    if is_polar:
        r_vec = jnp.linspace(R1, R2, grid_res)
        theta_vec = jnp.linspace(0, 2 * np.pi, grid_res)
        rr, tt = jnp.meshgrid(r_vec, theta_vec, indexing='ij')
        test_input = jnp.stack([rr.flatten(), tt.flatten()], axis=1)
        xx, yy = None, None
    else:
        x_vec = jnp.linspace(-R2, R2, grid_res)
        y_vec = jnp.linspace(-R2, R2, grid_res)
        xx, yy = jnp.meshgrid(x_vec, y_vec, indexing='ij')
        test_input = jnp.stack([xx.flatten(), yy.flatten()], axis=1)

    input1 = test_input[:, 0:1]
    input2 = test_input[:, 1:2]
    
    # 모델 파라미터 초기화 및 예측
    params = model.init(key, input1, input2)['params']
    output = model.apply({'params': params}, input1, input2) # output: (grid_res*grid_res, 2)
    
    # SVD를 위해 그리드 형태로 재구성
    sol1 = output[:, 0].reshape(grid_res, grid_res)
    sol2 = output[:, 1].reshape(grid_res, grid_res)

    # 직교좌표계의 경우, 원통 외부 영역은 0으로 마스킹
    if not is_polar:
        r_cart = jnp.sqrt(xx**2 + yy**2)
        mask = (r_cart < R1) | (r_cart > R2)
        sol1 = sol1.at[mask].set(0)
        sol2 = sol2.at[mask].set(0)
            
    return sol1, sol2

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 2)
u_low_res, v_low_res = get_low_res_solution_jax(is_polar=False, key=keys[0])
vr_low_res, vtheta_low_res = get_low_res_solution_jax(is_polar=True, key=keys[1])
print("저해상도 해 생성\n")

# --- SI 계산 및 최적 좌표계 선택 ---
print("--- 분리 가능성 지수(SI) 계산 ---")
si_u, s_u = calculate_si(u_low_res)
si_v, s_v = calculate_si(v_low_res)
si_vr, s_vr = calculate_si(vr_low_res)
si_vtheta, s_vtheta = calculate_si(vtheta_low_res)
print(f"직교좌표계 u(x,y)의 SI (저해상도): {si_u:.4f}")
print(f"직교좌표계 v(x,y)의 SI (저해상도): {si_v:.4f}")
print(f"극좌표계 v_r(r,θ)의 SI (저해상도):  {si_vr:.4f}")
print(f"극좌표계 v_θ(r,θ)의 SI (저해상도):  {si_vtheta:.4f}")

if max(si_vr, si_vtheta) > max(si_u, si_v):
    optimal_system = "극좌표계 (Polar)"
    optimal_s = s_vr if si_vr > si_vtheta else s_vtheta
    print("\n결론: 저해상도 해 분석 결과, 극좌표계가 최적 좌표계로 선택되었습니다.")
else:
    optimal_system = "직교좌표계 (Cartesian)"
    optimal_s = s_u if si_u > si_v else s_v
    print("\n결론: 저해상도 해 분석 결과, 직교좌표계가 최적 좌표계로 선택되었습니다.")

# --- 최소 유효 랭크(reff) 추정 ---
print("\n--- 3단계: 최소 유효 랭크(reff) 추정 ---")
reff, cumulative_energy = estimate_reff(optimal_s)
print(f"선택된 '{optimal_system}'에서,")
print(f"에너지의 99.9%를 보존하는 최소 유효 랭크(reff)는 {reff}로 추정됩니다.")


try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Dejavu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
except Exception as e:
    print(f"Font setting error: {e}. Trying 'Malgun Gothic'...")
    try:
         plt.rcParams['font.family'] = 'Malgun Gothic'
         plt.rcParams['axes.unicode_minus'] = False
    except:
         print("Font setting failed. Using default font.")
         pass

# --- Plot 1: Combined Singular Values ---
fig, ax = plt.subplots(1, 1, figsize=(6, 4)) # (width, height) in inches

if np.sum(s_u) > 1e-9:
    ax.plot(s_u / np.sum(s_u), 'o-', color='blue', markersize=4, linewidth=1.5, label='u(x,y) (Cartesian)')
if np.sum(s_v) > 1e-9:
    ax.plot(s_v / np.sum(s_v), 's--', color='blue', markersize=4, linewidth=1.5, label='v(x,y) (Cartesian)')

if np.sum(s_vr) > 1e-9:
    ax.plot(s_vr / np.sum(s_vr), 'o-', color='red', markersize=4, linewidth=1.5, label=r'$v_r(r,\theta)$ (Polar)')
if np.sum(s_vtheta) > 1e-9:
    ax.plot(s_vtheta / np.sum(s_vtheta), 's--', color='red', markersize=4, linewidth=1.5, label=r'$v_\theta(r,\theta)$ (Polar)')

ax.set_yscale('log')

ax.set_xlabel("Singular Value Index")
ax.set_ylabel("Normalized Singular Value")
ax.legend()
ax.grid(False)
ax.set_ylim(bottom=1e-6, top=2.0) 
ax.set_xlim(left=-1, right=grid_res)

plt.tight_layout()
plt.savefig("SPINN_Plot_A_SingularValues.png", dpi=300, bbox_inches='tight')
print("\n그래프 (A)를 'SPINN_Plot_A_SingularValues.png' (300 dpi) 파일 저장")
plt.show()


# --- Plot 2: Effective Rank (reff) Estimation ---
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(range(1, len(cumulative_energy) + 1), cumulative_energy, 'o-', color='black', markersize=4, linewidth=1.5, label='Cumulative Energy')
ax.axhline(y=0.999, color='r', linestyle='--', linewidth=1.5, label='99.9% Threshold')
ax.axvline(x=reff, color='g', linestyle='--', linewidth=1.5, label=f'reff = {reff}')

ax.set_xlabel("Rank (k)")
ax.set_ylabel("Cumulative Energy")

x_limit = max(10, reff + 5)
ax.set_xlim(0, x_limit)
ax.set_ylim(0, 1.05)

ax.legend(loc='lower right')
ax.grid(False) 

plt.tight_layout()
plt.savefig("SPINN_Plot_B_Reff.png", dpi=300, bbox_inches='tight')
print("\n그래프 (B)를 'SPINN_Plot_B_Reff.png' (300 dpi) 파일 저장")
plt.show()

print(f"   - 예측된 최적 좌표계: {optimal_system}")
print(f"  - 예측된 권장 랭크: {reff}")