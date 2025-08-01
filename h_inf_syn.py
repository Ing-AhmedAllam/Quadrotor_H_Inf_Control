import numpy as np
from scipy.linalg import solve
# from scipy.linalg import balance

def h_inf_norm(A, B, C, D):
    """
    Calculate the H-infinity norm of a system defined by matrices A, B, C, and D.
    
    :param A: State matrix
    :param B: Input matrix
    :param C: Output matrix
    :param D: Feedthrough matrix
    :return: H-infinity norm of the system
    """
    eps = 1e-6
    gamma_l = 0
    gamma_u = 1e10
    
    while abs(gamma_u - gamma_l) > eps:
        # bisection searching
        gamma = (gamma_l + gamma_u) / 2

        # construct Hamiltonian matrix
        # H = [A, (1/gamma)*(B @ B.T); -(1/gamma)*(C.T @ C), -A.T]
        top = np.hstack((A, (1/gamma) * (B @ B.conj().T)))
        bottom = np.hstack((- (1/gamma) * (C.T @ C), -A.T))
        H = np.vstack((top, bottom))
        
        # Check if H has pure imaginary eigenvalues
        eigvals = np.linalg.eigvals(H)
        has_pure_img = np.any(np.isclose(eigvals.real, 0, atol=1e-8) & (eigvals.imag != 0))
        
        if not has_pure_img:
            gamma_u = gamma  # decrease gamma upper bound
        else:
            gamma_l = gamma  # increase gamma lower bound
    
    return gamma

def care_sda(A: np.ndarray, H : np.ndarray, G : np.ndarray):
    fail = 0
    max_iter = 50
    state_dim = A.shape[0]

    r = 2.4  # SDA's author suggested the value between 2.1~2.6
    I = np.eye(state_dim)
    A_r = A - r * I
    A_r_inv = np.linalg.inv(A_r)
    A_hat_last = I + 2*r*np.linalg.inv(A_r + G@A_r_inv.T @ H)
    G_hat_last = 2*r*A_r_inv @ G @ np.linalg.inv(A_r.T + H @ A_r_inv @ G)
    # G_hat_last = 2*r*np.linalg.solve(A_r, G) @ np.linalg.inv(A_r.T + H @ A_r_inv @ G)
    H_hat_last = 2*r*np.linalg.inv(A_r.T + H @ A_r_inv @ G) @ H @ A_r_inv

    residual_now = np.linalg.norm(H_hat_last)

    iter = 0
    
    while iter < max_iter:
        iter += 1
        # reduce redundent calculation by pre-calculating repeated terms
        I_plus_H_G = I + (H_hat_last @ G_hat_last)
        A_hat_last_t = A_hat_last.T
        
        #update
        # cond = np.linalg.cond(I_plus_H_G)
        # if cond > 1e10:
        #     print("High condition number:", cond)
        #     print("I_plus_H_G:", I_plus_H_G)
        #     exit(1)
        # scale = np.linalg.norm(I_plus_H_G, ord='fro')  # or np.max(np.abs(I_plus_H_G))
        # epsilon = 1e-6*scale
        # I_plus_H_G_reg = I_plus_H_G + epsilon * np.eye(I_plus_H_G.shape[0])
        # mat_to_invert = I + G_hat_last @ H_hat_last + epsilon * np.eye(state_dim)

        # A_hat_new = A_hat_last @ solve(mat_to_invert, A_hat_last)
        # G_hat_new = G_hat_last + A_hat_last @ G_hat_last @ solve(I_plus_H_G_reg, A_hat_last_t)
        # H_hat_new = H_hat_last + A_hat_last_t @ solve(I_plus_H_G_reg, H_hat_last @ A_hat_last)
        A_hat_new = A_hat_last @ np.linalg.inv(I+ G_hat_last @ H_hat_last) @ A_hat_last
        G_hat_new = G_hat_last + A_hat_last @ G_hat_last @ np.linalg.inv(I_plus_H_G) @ A_hat_last_t
        H_hat_new = H_hat_last + A_hat_last_t @ np.linalg.inv(I_plus_H_G) @ H_hat_last @ A_hat_last

        #matrix norms
        residual_last = residual_now
        residual_now = np.linalg.norm(H_hat_new - H_hat_last)

        #prepare next iteration
        A_hat_last = A_hat_new
        G_hat_last = G_hat_new
        H_hat_last = H_hat_new

        #diverge
        if iter >= max_iter:
            X = []
            fail = 1
            return X, fail

        #stop iteration if converged
        if residual_now < 1e-10:
            break
    X = H_hat_new
    
    return X, fail

def h_inf_syn(A, B1, B2, C1, D):
    """a bisection method to sythesis the H-infinity control solution with
        minimal gamma using the structure-preserving doubling algorithm
    """
    eps = 1e-6  # Small value to avoid division by zero
    residual_eps = 1e-7  # Residual error threshold
    
    iter = 0
    
    At = A.T
    B1sqt = B1 @ B1.T
    B2sqt = B2 @ B2.T
    C1tsqt = C1.T @ C1
    
    # Calculate the lower bound gamma
    H = B2sqt
    G = C1tsqt
    
    Z, fail = care_sda(At, H, G)
    gamma_lb = h_inf_norm((A-Z@C1tsqt).T, C1.T, B1.T, 0)
    gamma_l = gamma_lb
    
    # approximate an upper bound gamma:
    # MATLAB's eigs(B1'*B1,1,'lm') finds the largest eigenvalue
    # In numpy, use np.linalg.eigvalsh for symmetric matrices
    rho_max = np.max(np.linalg.eigvalsh(B1.T @ B1))  # largest eigenvalue of B1.T @ B1
    rho_0 = np.min(np.linalg.eigvalsh(B2.T @ B2))    # smallest eigenvalue of B2.T @ B2
    
    gamma_u = rho_max / rho_0
    
    optimal_residual = np.inf
    optimal_gamma = gamma_u
    optimal_X = []
    
    while True:
        iter += 1
        
        gamma = (gamma_l + gamma_u) / 2
        H = C1tsqt
        G = B2sqt - (B1sqt / gamma**2)

        r1 = np.hstack((A, -G))
        r2 = np.hstack((-H, -At))
        Ham = np.vstack((r1, r2))
        
        # Ham,_ = balance(Ham)
        # Ham = np.array(Ham)
        
        # Check if H has pure imaginary eigenvalues
        eigvals = np.linalg.eigvals(Ham)
        has_pure_img = np.any(np.isclose(eigvals.real, 0, atol=1e-8) & (eigvals.imag != 0))
        
        if not has_pure_img:
            X, fail = care_sda(A, H, G)
            
            if fail == 0:
                #check if the solution is stable
                ric_residual = np.linalg.norm(At@X + X@A - X@G@X + H)
                if ric_residual < residual_eps:
                    optimal_residual = ric_residual
                    optimal_gamma = gamma
                    optimal_X = X
                    
                    gamma_u = gamma  # decrease gamma upper bound
                else:
                    gamma_l = gamma
            else:
                gamma_l = gamma
        else:
            gamma_l = gamma
        
        if abs(gamma_u - gamma_l) < eps:
            if optimal_residual < residual_eps:
                return optimal_gamma, gamma_lb, optimal_X, optimal_residual
            else:
                G = B2sqt
                X, fail = care_sda(A, H, G)
                ric_residual = np.linalg.norm(At@X + X@A - X@G@X + H)
                gamma = 0
                return gamma, gamma_lb, X, ric_residual
            
        
        
        
        
