import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import jax
import jax.numpy as jnp
import optax
from functools import partial
from jax import jvp, grad
import matplotlib.cm as cm
from networks.physics_informed_neural_networks import PINN2d

@partial(jax.jit, static_argnums=(0,))
def hvp_fwdfwd(fn, primals, tangents):
    if isinstance(primals, (list, tuple)):
        primals = primals[0].flatten() 
    else:
        primals = primals.flatten()
        
    if isinstance(tangents, (list, tuple)):
        tangents = tangents[0].flatten()
    else:
        tangents = tangents.flatten()

    g = grad(lambda x: fn(x[:, None]).sum())(primals) 
    return jvp(lambda x: grad(lambda x: fn(x[:, None]).sum())(x), (primals,), (tangents,))[1][:, None] 


def calculate_si(matrix):
    matrix_np = np.array(matrix)
    matrix_np = np.nan_to_num(matrix_np) 
    singular_values = svd(matrix_np, compute_uv=False)
    if np.sum(singular_values) < 1e-9: return 0.0, singular_values
    si = singular_values[0] / np.sum(singular_values)
    return si, singular_values

def estimate_reff(singular_values, threshold=0.999):
    if np.sum(singular_values) < 1e-9: return 0, np.array([0])
    normalized_s = singular_values / np.sum(singular_values)
    cumulative_energy = np.cumsum(normalized_s)
    reff = np.searchsorted(cumulative_energy, threshold, side='right') + 1
    return reff, cumulative_energy

R1, R2 = 1.0, 2.0
omega1, omega2 = 1.0, 0.0
nu = 0.1 
lbda_b = 100.0
lbda_data = 100.0 
rho = 1.0 
A = (omega1*R1**2 - omega2*R2**2) / (R1**2 - R2**2)
B = (omega2 - omega1) * R1**2 * R2**2 / (R1**2 - R2**2)
grid_res = 50 

@jax.jit
def get_analytical_solution_polar(r, theta):
    v_theta = A * r + B / r
    v_r = jnp.zeros_like(r) 
    return v_r, v_theta

@jax.jit
def get_analytical_solution_cartesian(x, y):
    r = jnp.sqrt(x**2 + y**2)
    omega = A + B / (r**2)
    v_x = -omega * y
    v_y = omega * x
    return v_x, v_y

@partial(jax.jit, static_argnums=(0,))
def loss_fn_low_cost_polar(apply_fn, params, rc, thetac, rb, thetab):
    vr_b, vtheta_b = get_analytical_solution_polar(rb, thetab) 

    @jax.jit
    def model_output(r, theta):
        outputs = apply_fn({'params': params}, r, theta)
        return outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3] 

    def residual_loss(params, r, theta):
        def v_r_scalar(r_s, theta_s):
            return apply_fn({'params': params}, jnp.array([[r_s]]), jnp.array([[theta_s]]))[0, 0]
        def v_theta_scalar(r_s, theta_s):
            return apply_fn({'params': params}, jnp.array([[r_s]]), jnp.array([[theta_s]]))[0, 1]
        def p_scalar(r_s, theta_s):
            return apply_fn({'params': params}, jnp.array([[r_s]]), jnp.array([[theta_s]]))[0, 2]

        r_flat = r.flatten()
        theta_flat = theta.flatten()

        vr_vec = jax.vmap(v_r_scalar)(r_flat, theta_flat)[:, None]
        vt_vec = jax.vmap(v_theta_scalar)(r_flat, theta_flat)[:, None]
        
        vr_r_vec = jax.vmap(grad(v_r_scalar, argnums=0))(r_flat, theta_flat)[:, None]
        vr_t_vec = jax.vmap(grad(v_r_scalar, argnums=1))(r_flat, theta_flat)[:, None]
        vt_r_vec = jax.vmap(grad(v_theta_scalar, argnums=0))(r_flat, theta_flat)[:, None]
        vt_t_vec = jax.vmap(grad(v_theta_scalar, argnums=1))(r_flat, theta_flat)[:, None]
        p_r_vec = jax.vmap(grad(p_scalar, argnums=0))(r_flat, theta_flat)[:, None]
        p_t_vec = jax.vmap(grad(p_scalar, argnums=1))(r_flat, theta_flat)[:, None]

        vr_rr_vec = jax.vmap(grad(grad(v_r_scalar, argnums=0), argnums=0))(r_flat, theta_flat)[:, None]
        vr_tt_vec = jax.vmap(grad(grad(v_r_scalar, argnums=1), argnums=1))(r_flat, theta_flat)[:, None]
        vt_rr_vec = jax.vmap(grad(grad(v_theta_scalar, argnums=0), argnums=0))(r_flat, theta_flat)[:, None]
        vt_tt_vec = jax.vmap(grad(grad(v_theta_scalar, argnums=1), argnums=1))(r_flat, theta_flat)[:, None]

        mu = nu * rho
        r_col = r_flat[:, None] 

        e1 = vr_r_vec + vr_vec / r_col + vt_t_vec / r_col

        term_inertial_r = rho * (vr_vec * vr_r_vec + (vt_vec / r_col) * vr_t_vec - (vt_vec**2 / r_col))
        term_pressure_r = p_r_vec
        term_viscous_r = mu * (vr_rr_vec + (1.0 / r_col) * vr_r_vec + (1.0 / r_col**2) * vr_tt_vec - vr_vec / r_col**2 - (2.0 / r_col**2) * vt_t_vec)
        e2 = term_inertial_r + term_pressure_r - term_viscous_r

        term_inertial_t = rho * (vr_vec * vt_r_vec + (vt_vec / r_col) * vt_t_vec + (vr_vec * vt_vec) / r_col)
        term_pressure_t = (1.0 / r_col) * p_t_vec
        term_viscous_t = mu * (vt_rr_vec + (1.0 / r_col) * vt_r_vec + (1.0 / r_col**2) * vt_tt_vec - vt_vec / r_col**2 + (2.0 / r_col**2) * vr_t_vec)
        e3 = term_inertial_t + term_pressure_t - term_viscous_t

        loss_e1 = jnp.mean(e1**2)
        loss_e2 = jnp.mean(e2**2)
        loss_e3 = jnp.mean(e3**2)
        
        return loss_e1 + loss_e2 + loss_e3

    def boundary_loss(params, r_b, theta_b, vr_b, vtheta_b):
        vr_pred, vtheta_pred, _ = model_output(r_b, theta_b)
        
        loss_vr = jnp.mean((vr_pred - vr_b)**2)
        loss_vtheta = jnp.mean((vtheta_pred - vtheta_b)**2)
        return loss_vr + loss_vtheta

    loss_r = residual_loss(params, rc, thetac)
    loss_b = boundary_loss(params, rb, thetab, vr_b, vtheta_b)
    
    total_loss = loss_r + lbda_b * loss_b
    return total_loss

@partial(jax.jit, static_argnums=(0,))
def loss_fn_low_cost_cartesian(apply_fn, params, xc, yc, xb, yb): 

    u_b, v_b = get_analytical_solution_cartesian(xb, yb)

    @jax.jit
    def model_output(x, y):
        outputs = apply_fn({'params': params}, x, y)
        return outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3] 

    def residual_loss(params, x, y):
        
        def u_scalar(x_s, y_s):
            return apply_fn({'params': params}, jnp.array([[x_s]]), jnp.array([[y_s]]))[0, 0]
        def v_scalar(x_s, y_s):
            return apply_fn({'params': params}, jnp.array([[x_s]]), jnp.array([[y_s]]))[0, 1]
        def p_scalar(x_s, y_s):
            return apply_fn({'params': params}, jnp.array([[x_s]]), jnp.array([[y_s]]))[0, 2]

        u_vec = jax.vmap(u_scalar)(x.flatten(), y.flatten())[:, None]
        v_vec = jax.vmap(v_scalar)(x.flatten(), y.flatten())[:, None]
        
        u_x_vec = jax.vmap(grad(u_scalar, argnums=0))(x.flatten(), y.flatten())[:, None]
        u_y_vec = jax.vmap(grad(u_scalar, argnums=1))(x.flatten(), y.flatten())[:, None]
        v_x_vec = jax.vmap(grad(v_scalar, argnums=0))(x.flatten(), y.flatten())[:, None]
        v_y_vec = jax.vmap(grad(v_scalar, argnums=1))(x.flatten(), y.flatten())[:, None]
        p_x_vec = jax.vmap(grad(p_scalar, argnums=0))(x.flatten(), y.flatten())[:, None]
        p_y_vec = jax.vmap(grad(p_scalar, argnums=1))(x.flatten(), y.flatten())[:, None]
        
        u_xx_vec = jax.vmap(grad(grad(u_scalar, argnums=0), argnums=0))(x.flatten(), y.flatten())[:, None]
        u_yy_vec = jax.vmap(grad(grad(u_scalar, argnums=1), argnums=1))(x.flatten(), y.flatten())[:, None]
        v_xx_vec = jax.vmap(grad(grad(v_scalar, argnums=0), argnums=0))(x.flatten(), y.flatten())[:, None]
        v_yy_vec = jax.vmap(grad(grad(v_scalar, argnums=1), argnums=1))(x.flatten(), y.flatten())[:, None]

        mu = nu * rho 
        
        e1 = u_x_vec + v_y_vec
        e2 = rho * (u_vec * u_x_vec + v_vec * u_y_vec) + p_x_vec - mu * (u_xx_vec + u_yy_vec)
        e3 = rho * (u_vec * v_x_vec + v_vec * v_y_vec) + p_y_vec - mu * (v_xx_vec + v_yy_vec)
        
        loss_e1 = jnp.mean(e1**2)
        loss_e2 = jnp.mean(e2**2)
        loss_e3 = jnp.mean(e3**2)
        
        return loss_e1 + loss_e2 + loss_e3

    def boundary_loss(params, x_b, y_b, u_b, v_b):
        u_pred, v_pred, _ = model_output(x_b, y_b)
        
        loss_u = jnp.mean((u_pred - u_b)**2)
        loss_v = jnp.mean((v_pred - v_b)**2)
        return loss_u + loss_v

    loss_r = residual_loss(params, xc, yc)
    loss_b = boundary_loss(params, xb, yb, u_b, v_b)
    
    total_loss = loss_r + lbda_b * loss_b 
    return total_loss

def get_low_res_solution_jax(is_polar, key):
    
    N_low_cost_epochs = 2000 
    N_b = 100           
    N_c = 500
    lr = 1e-3

    key, model_key, data_key = jax.random.split(key, 3)

    if is_polar:
        print("--- Polar 저비용 PINN (PDE + BC) 학습 시작 ---")
        low_cost_features = [16, 16, 3] 
        model = PINN2d(features=low_cost_features)
        
        key_b1, key_b2, key_rc, key_tc = jax.random.split(data_key, 4)
        
        theta_b1 = jax.random.uniform(key_b1, (N_b//2, 1)) * 2 * jnp.pi
        r_b1 = jnp.ones_like(theta_b1) * R1
        theta_b2 = jax.random.uniform(key_b2, (N_b//2, 1)) * 2 * jnp.pi
        r_b2 = jnp.ones_like(theta_b2) * R2
        rb = jnp.concatenate([r_b1, r_b2])
        thetab = jnp.concatenate([theta_b1, theta_b2])
        r_c = jnp.sqrt(jax.random.uniform(key_rc, (N_c, 1)) * (R2**2 - R1**2) + R1**2)
        theta_c = jax.random.uniform(key_tc, (N_c, 1)) * 2 * jnp.pi
        rc = r_c
        thetac = theta_c
        
        params = model.init(model_key, rc, thetac)['params']
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        loss_grad_fn_polar = jax.value_and_grad(
            partial(loss_fn_low_cost_polar, model.apply)
        )

        @jax.jit
        def train_step_polar(params, opt_state):
            loss_val, grads = loss_grad_fn_polar(params, rc, thetac, rb, thetab)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        for e in range(N_low_cost_epochs): 
            params, opt_state, loss = train_step_polar(params, opt_state)
            if e % 500 == 0:
                print(f"Polar Epoch {e}: Loss {loss:.6f}")
        print("--- Polar 저비용 PINN 학습 완료 ---")

        r_vec = jnp.linspace(R1, R2, grid_res)
        theta_vec = jnp.linspace(0, 2 * jnp.pi, grid_res)
        rr, tt = jnp.meshgrid(r_vec, theta_vec, indexing='ij')
        
        rr_flat = rr.flatten()[:, None]
        tt_flat = tt.flatten()[:, None]
        output = model.apply({'params': params}, rr_flat, tt_flat)
        
        sol1 = output[:, 0].reshape(grid_res, grid_res) 
        sol2 = output[:, 1].reshape(grid_res, grid_res) 
        return sol1, sol2, rr, tt 

    else:
        print("--- Cartesian 저비용 PINN (PDE + BC) 학습 시작 ---")
        
        low_cost_features = [16, 16, 3] 
        model = PINN2d(features=low_cost_features)

        key_b1, key_b2, key_rc, key_tc = jax.random.split(data_key, 4)

        theta_b1 = jax.random.uniform(key_b1, (N_b//2, 1)) * 2 * jnp.pi
        theta_b2 = jax.random.uniform(key_b2, (N_b//2, 1)) * 2 * jnp.pi
        xb1 = R1 * jnp.cos(theta_b1)
        yb1 = R1 * jnp.sin(theta_b1)
        xb2 = R2 * jnp.cos(theta_b2)
        yb2 = R2 * jnp.sin(theta_b2)
        xb = jnp.concatenate([xb1, xb2])
        yb = jnp.concatenate([yb1, yb2])
        
        r_c = jnp.sqrt(jax.random.uniform(key_rc, (N_c, 1)) * (R2**2 - R1**2) + R1**2)
        theta_c = jax.random.uniform(key_tc, (N_c, 1)) * 2 * jnp.pi
        xc = r_c * jnp.cos(theta_c)
        yc = r_c * jnp.sin(theta_c)
        
        params = model.init(model_key, xc, yc)['params']
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        
        loss_grad_fn_cart = jax.value_and_grad(
            partial(loss_fn_low_cost_cartesian, model.apply)
        )

        @jax.jit
        def train_step_cartesian(params, opt_state):
            loss_val, grads = loss_grad_fn_cart(params, xc, yc, xb, yb) 
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val
        
        x_vec_eval = jnp.linspace(-R2, R2, grid_res)
        y_vec_eval = jnp.linspace(-R2, R2, grid_res)
        xx, yy = jnp.meshgrid(x_vec_eval, y_vec_eval, indexing='ij')

        for e in range(N_low_cost_epochs):
            params, opt_state, loss = train_step_cartesian(params, opt_state)
            if e % 500 == 0:
                loss_val, _ = loss_grad_fn_cart(params, xc, yc, xb, yb) 
                print(f"Cartesian Epoch {e}: Loss {loss_val:.6f}")
        
        loss_val, _ = loss_grad_fn_cart(params, xc, yc, xb, yb) 
        print(f"Cartesian Epoch {N_low_cost_epochs}: Loss {loss_val:.6f}")

        print("--- Cartesian 저비용 PINN 학습 완료 ---")

        xx_flat = xx.flatten()[:, None]
        yy_flat = yy.flatten()[:, None]
        output = model.apply({'params': params}, xx_flat, yy_flat)
        
        sol1 = output[:, 0].reshape(grid_res, grid_res) 
        sol2 = output[:, 1].reshape(grid_res, grid_res) 

        r_cart = jnp.sqrt(xx**2 + yy**2)
        mask = (r_cart < R1) | (r_cart > R2)
        
        sol1 = sol1.at[mask].set(jnp.nan)
        sol2 = sol2.at[mask].set(jnp.nan)
            
        return sol1, sol2, xx, yy

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 2)

u_low_res, v_low_res, xx, yy = get_low_res_solution_jax(is_polar=False, key=keys[0])
vr_low_res, vtheta_low_res, rr, tt = get_low_res_solution_jax(is_polar=True, key=keys[1]) 

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

vx_exact, vy_exact = get_analytical_solution_cartesian(xx, yy)

r_cart_exact_plot = jnp.sqrt(xx**2 + yy**2)
mask_exact_plot = (r_cart_exact_plot < R1) | (r_cart_exact_plot > R2)
vx_exact = vx_exact.at[mask_exact_plot].set(jnp.nan)
vy_exact = vy_exact.at[mask_exact_plot].set(jnp.nan)

vx_error = u_low_res - vx_exact
vy_error = v_low_res - vy_exact

fig_c, axes_c = plt.subplots(2, 3, figsize=(15, 8))

cmap_jet_nan_c = cm.jet.copy()
cmap_jet_nan_c.set_bad(color='lightgray')
cmap_coolwarm_nan_c = cm.coolwarm.copy()
cmap_coolwarm_nan_c.set_bad(color='lightgray')

v_min, v_max = jnp.nanmin(vx_exact), jnp.nanmax(vx_exact) 
im1 = axes_c[0, 0].imshow(u_low_res, cmap=cmap_jet_nan_c, aspect='auto', origin='lower', extent=[-R2, R2, -R2, R2], vmin=v_min, vmax=v_max)
axes_c[0, 0].set_title(r"$v_x$ (Approx.)")
fig_c.colorbar(im1, ax=axes_c[0, 0])
im2 = axes_c[0, 1].imshow(vx_exact, cmap=cmap_jet_nan_c, aspect='auto', origin='lower', extent=[-R2, R2, -R2, R2], vmin=v_min, vmax=v_max)
axes_c[0, 1].set_title(r"$v_x$ (Exact)")
fig_c.colorbar(im2, ax=axes_c[0, 1])
im3 = axes_c[0, 2].imshow(vx_error, cmap=cmap_coolwarm_nan_c, aspect='auto', origin='lower', extent=[-R2, R2, -R2, R2])
axes_c[0, 2].set_title(r"$v_x$ (Error)")
fig_c.colorbar(im3, ax=axes_c[0, 2])

v_min, v_max = jnp.nanmin(vy_exact), jnp.nanmax(vy_exact) 
im4 = axes_c[1, 0].imshow(v_low_res, cmap=cmap_jet_nan_c, aspect='auto', origin='lower', extent=[-R2, R2, -R2, R2], vmin=v_min, vmax=v_max)
axes_c[1, 0].set_title(r"$v_y$ (Approx.)")
fig_c.colorbar(im4, ax=axes_c[1, 0])
im5 = axes_c[1, 1].imshow(vy_exact, cmap=cmap_jet_nan_c, aspect='auto', origin='lower', extent=[-R2, R2, -R2, R2], vmin=v_min, vmax=v_max)
axes_c[1, 1].set_title(r"$v_y$ (Exact)")
fig_c.colorbar(im5, ax=axes_c[1, 1])
im6 = axes_c[1, 2].imshow(vy_error, cmap=cmap_coolwarm_nan_c, aspect='auto', origin='lower', extent=[-R2, R2, -R2, R2])
axes_c[1, 2].set_title(r"$v_y$ (Error)")
fig_c.colorbar(im6, ax=axes_c[1, 2])

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("SPINN_Plot_Solutions_Cartesian.png", dpi=300)

vr_exact, vtheta_exact = get_analytical_solution_polar(rr, tt)
vr_error = vr_low_res - vr_exact
vtheta_error = vtheta_low_res - vtheta_exact

fig_d, axes_d = plt.subplots(2, 3, figsize=(15, 8))

v_min_vr, v_max_vr = -0.1, 0.1 
im1 = axes_d[0, 0].imshow(vr_low_res, cmap=cm.jet, aspect='auto', origin='lower', extent=[R1, R2, 0, 2*jnp.pi], vmin=v_min_vr, vmax=v_max_vr)
axes_d[0, 0].set_title(r"$v_r$ (Approx.)")
fig_d.colorbar(im1, ax=axes_d[0, 0])
im2 = axes_d[0, 1].imshow(vr_exact, cmap=cm.jet, aspect='auto', origin='lower', extent=[R1, R2, 0, 2*jnp.pi], vmin=v_min_vr, vmax=v_max_vr)
axes_d[0, 1].set_title(r"$v_r$ (Exact)")
fig_d.colorbar(im2, ax=axes_d[0, 1])
im3 = axes_d[0, 2].imshow(vr_error, cmap='coolwarm', aspect='auto', origin='lower', extent=[R1, R2, 0, 2*jnp.pi])
axes_d[0, 2].set_title(r"$v_r$ (Error)")
fig_c.colorbar(im3, ax=axes_d[0, 2])

v_min_vt, v_max_vt = jnp.min(vtheta_exact), jnp.max(vtheta_exact)
im4 = axes_d[1, 0].imshow(vtheta_low_res, cmap=cm.jet, aspect='auto', origin='lower', extent=[R1, R2, 0, 2*jnp.pi], vmin=v_min_vt, vmax=v_max_vt)
axes_d[1, 0].set_title(r"$v_\theta$ (Approx.)")
fig_d.colorbar(im4, ax=axes_d[1, 0])
im5 = axes_d[1, 1].imshow(vtheta_exact, cmap=cm.jet, aspect='auto', origin='lower', extent=[R1, R2, 0, 2*jnp.pi], vmin=v_min_vt, vmax=v_max_vt)
axes_d[1, 1].set_title(r"$v_\theta$ (Exact)")
fig_d.colorbar(im5, ax=axes_d[1, 1]) 
im6 = axes_d[1, 2].imshow(vtheta_error, cmap='coolwarm', aspect='auto', origin='lower', extent=[R1, R2, 0, 2*jnp.pi])
axes_d[1, 2].set_title(r"$v_\theta$ (Error)")
fig_d.colorbar(im6, ax=axes_d[1, 2])

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("SPINN_Plot_Solutions_Polar.png", dpi=300)

si_u, s_u = calculate_si(u_low_res)
si_v, s_v = calculate_si(v_low_res)
si_vr, s_vr = calculate_si(vr_low_res)
si_vtheta, s_vtheta = calculate_si(vtheta_low_res)
print(f"직교좌표계 u(x,y)의 SI (PINN 근사해): {si_u:.4f}")
print(f"직교좌표계 v(x,y)의 SI (PINN 근사해): {si_v:.4f}")
print(f"극좌표계 v_r(r,θ)의 SI (PINN 근사해):     {si_vr:.4f}")
print(f"극좌표계 v_θ(r,θ)의 SI (PINN 근사해):     {si_vtheta:.4f}")

si_polar_max = si_vtheta 
si_cartesian_max = max(si_u, si_v)

if si_polar_max > si_cartesian_max: 
    optimal_system = "극좌표계 (Polar)"
    optimal_s = s_vtheta 
    print("\n결론: 저주파 근사해 분석 결과, 극좌표계가 최적 좌표계로 선택되었습니다.")
else:
    optimal_system = "직교좌표계 (Cartesian)"
    optimal_s = s_u if si_u > si_v else s_v
    print("\n결론: 저주파 근사해 분석 결과, 직교좌표계가 최적 좌표계로 선택되었습니다.")
reff, cumulative_energy = estimate_reff(optimal_s)
print(f"선택된 '{optimal_system}'에서,")
print(f"에너지의 99.9%를 보존하는 최소 유효 랭크(reff)는 {reff}로 추정됩니다.")

fig, ax = plt.subplots(1, 1, figsize=(6, 4)) 

if np.sum(s_u) > 1e-9:
    ax.plot(s_u / np.sum(s_u), 'o-', color='blue', markersize=4, linewidth=1.5, label='u(x,y) (Cartesian)')
if np.sum(s_v) > 1e-9:
    ax.plot(s_v / np.sum(s_v), 's--', color='blue', markersize=4, linewidth=1.5, label='v(x,y) (Cartesian)')
if np.sum(s_vr) > 1e-9:
    ax.plot(s_vr / np.sum(s_vr), 's--', color='red', markersize=4, linewidth=1.5, label=r'$v_r(r,\theta)$ (Polar)')
if np.sum(s_vtheta) > 1e-9:
    ax.plot(s_vtheta / np.sum(s_vtheta), 'o-', color='red', markersize=4, linewidth=1.5, label=r'$v_\theta(r,\theta)$ (Polar)')

ax.set_yscale('log')
ax.set_xlabel("Singular Value Index")
ax.set_ylabel("Normalized Singular Value")
ax.legend()
ax.grid(False)
ax.set_ylim(bottom=1e-6, top=2.0) 
ax.set_xlim(left=-1, right=grid_res)

plt.tight_layout()
plt.savefig("SPINN_Plot_A_SingularValues.png", dpi=300, bbox_inches='tight')
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(range(1, len(cumulative_energy) + 1), cumulative_energy, 'o-', color='black', markersize=4, linewidth=1.5, label='Cumulative Energy')
ax.axhline(y=0.999, color='r', linestyle='--', linewidth=1.5, label='99.9% Threshold')
ax.axvline(x=reff, color='g', linestyle='--', linewidth=1.5, label=f'reff = {reff}')

ax.set_xlabel("Rank (k)")
ax.set_ylabel("Cumulative Energy")
x_limit = max(10, reff + 5)
ax.set_xlim(0, x_limit)
ax.set_ylim(0.9, 1.01) 

ax.legend(loc='lower right')
ax.grid(False) 

plt.tight_layout()
plt.savefig("SPINN_Plot_B_Reff.png", dpi=300, bbox_inches='tight')

print(f"   - 예측된 최적 좌표계: {optimal_system}")
print(f"   - 예측된 권장 랭크: {reff}")