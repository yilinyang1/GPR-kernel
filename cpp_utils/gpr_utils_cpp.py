from ase.db import connect
import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import cholesky, det
from scipy.optimize import minimize
from ase.calculators.calculator import (Calculator, all_changes, PropertyNotImplementedError)
from ase.optimize import BFGS
from math import sqrt
from .gen_ffi import _gen_2Darray_for_ffi
from ._libkernel import lib, ffi
import time


# three types of kernel: energy-energy, energy-force, force-force
def ee_kernel(x, xp, lens):
    """
    x: (d, ), positions, row
    xp: (d, ), positions, col
    lens: (d, ), length scales
    """
    inner = 0.5 * (np.square((x - xp) / lens)).sum()
    return 1.0 * np.exp(-inner)

def ef_kernel(x, xp, lens, d):
    """
    x: (d, ), positions, row
    xp: (d, ), positions, col
    lens: (d, ), length scales
    d: which dimension of force or cartisian coordinate
    """
    pre = 1.0 * ((x[d] - xp[d])) / (lens**2)
    inner = 0.5 * (np.square((x - xp) / lens)).sum()
    return pre * np.exp(-inner)

def fe_kernel(x, xp, lens, d):
    """
    x: (d, ), positions, row
    xp: (d, ), positions, col
    lens: (d, ), length scales
    d: which dimension of force or cartisian coordinate
    """
    pre = -1.0 * ((x[d] - xp[d])) / (lens**2)
    inner = 0.5 * (np.square((x - xp) / lens)).sum()
    return pre * np.exp(-inner)

def ff_kernel(x, xp, lens, d, dp):
    """
    x: (d, ), positions, row
    xp: (d, ), positions, col
    lens: (d, ), length scales
    d: which dimension of force or cartisian coordinate
    dp: which dimension of force or cartisian coordinate
    """
    delta = 1 if d == dp else 0
    pre = 1.0 / (lens ** 2) * (delta - (x[d] - xp[d]) * (x[dp] - xp[dp]) / (lens ** 2))
    inner = 0.5 * (np.square((x - xp) / lens)).sum()
    return pre * np.exp(-inner)


def atoms2data(images):
    X = []
    y = []
    for atoms in images:
        entry_X = atoms.positions.flatten()
        entry_y_nrg = atoms.get_potential_energy()
        entry_y_frs = -1 * atoms.get_forces().flatten()
        entry_y = np.concatenate([[entry_y_nrg], entry_y_frs], axis=0)
        X.append(entry_X)
        y.append(entry_y)
    
    y_new = []  # reorder y
    n, d = len(images), 1 + len(images[0]) * 3
    for i in range(d):
        for j in range(n):
            y_new.append(y[j][i])
    return np.array(X), np.array(y_new)


def nll_obs(K, y, sigma_e, sigma_f, n):
    """
    K: kernel matrix of the training set
    y: observation of the training set
    sigma_e: noise of energy
    sigma_f: noise of forces
    n: number of training instances
    """
    diag = np.zeros_like(K)
    for i in range(n):
        diag[i][i] = sigma_e ** 2
    for i in range(n, len(diag)):
        diag[i][i] = sigma_f ** 2
    
    noise_K = K + diag
    
    # stable
    L = cholesky(noise_K)
    S1 = solve_triangular(L, y, lower=True)
    S2 = solve_triangular(L.T, S1, lower=False)
    nll = np.sum(np.log(np.diagonal(L))) + 0.5 * y.dot(S2) + 0.5 * n * np.log(2*np.pi)
    return nll


def cpp_kernel_train(X_train, lens):
    X_train_C = np.copy(X_train, order='C')
    X_train_Cp = _gen_2Darray_for_ffi(X_train_C, ffi)

    n, dim = X_train.shape[0], X_train.shape[1]
    res_kernels = np.zeros([n * (1 + dim), n * (1 + dim)], dtype=np.float64, order='C')
    res_kernelsp = _gen_2Darray_for_ffi(res_kernels, ffi)
    errno = lib.kernel_train(X_train_Cp, lens, n, dim, res_kernelsp)
    return res_kernels


def gpr_train(X_train, y_train, l_max):
    """
    X_train: [n, 3 * d]
    y_train: [n * (1 + 3d)]
    """    
    def obj_func(theta):
        sigma_e, sigma_f, lens = theta[0], theta[1], theta[2]
        kernel_matrix = cpp_kernel_train(X_train, lens)
        return nll_obs(kernel_matrix, y_train, sigma_e, sigma_f, len(X_train))

    init_sigma_e, init_sigma_f, init_lens = 0.005, 0.0005, 1.0
    init_theta = [init_sigma_e, init_sigma_f, init_lens] 
    # large lengscale range, old: [0.01, 0.1]
    bnds = [[0.001, 0.01], [0.0001, 0.001], [0.01, l_max]]
    res = minimize(obj_func, init_theta, bounds=bnds)
    return res


def gpr_predict(X_train, y_train, x_test, theta):
    sigma_e, sigma_f, lens = theta[0], theta[1], theta[2]
    n_train, dim = len(X_train), X_train.shape[1]
    kernel_matrix = cpp_kernel_train(X_train, lens)
    diag = np.zeros_like(kernel_matrix)
    for i in range(n_train):
        diag[i][i] = sigma_e ** 2
    for i in range(n_train, len(diag)):
        diag[i][i] = sigma_f ** 2
    noise_kernel = kernel_matrix + diag
    
    K_test_test = cpp_kernel_train(x_test.reshape(1, -1), lens)  # [1+d, 1+d]
    
    # predict energy
    K_eX = np.zeros(len(kernel_matrix[0]))
    # fill energy-energy
    for i in range(n_train):
        K_eX[i] = ee_kernel(x_test, X_train[i], lens)
    # fill energy-force
    for i in range(n_train, n_train * (1 + dim)):
        train_id = i % n_train
        d = i // n_train - 1
        K_eX[i] = ef_kernel(x_test, X_train[train_id], lens, d)
    
    # K_Xe = np.zeros(len(kernel_matrix))
    # # fill energy-energy
    # for i in range(n_train):
    #     K_Xe[i] = ee_kernel(X_train[i], x_test, lens)
    # # fill force energy
    # for i in range(n_train, n_train * (1 + dim)):
    #     train_id = i % n_train
    #     d = i // n_train - 1
    #     K_Xe[i] = fe_kernel(X_train[train_id], x_test, lens, d)
    
    mu_nrg = K_eX @ np.linalg.inv(noise_kernel) @ y_train
    var_nrg = K_test_test[0][0] - K_eX @ np.linalg.inv(noise_kernel) @ K_eX.T
    
    # predict forces
    K_fX = np.zeros([dim, len(kernel_matrix[0])])
    # fill force-energy
    for i in range(dim):
        for j in range(n_train):
            K_fX[i][j] = fe_kernel(x_test, X_train[j], lens, i)
    # fill force-force
    for i in range(dim):
        for j in range(n_train, n_train * (1 + dim)):
            train_id = j % n_train
            train_d = j // n_train - 1
            K_fX[i][j] = ff_kernel(x_test, X_train[train_id], lens, i, train_d)

    # # fill energy-force
    # K_Xf = np.zeros([len(kernel_matrix), dim])
    # for i in range(n_train):
    #     for j in range(dim):
    #         K_Xf[i][j] = ef_kernel(X_train[i], x_test, lens, j)
    # # fill force-force
    # for i in range(n_train, n_train * (1 + dim)):
    #     for j in range(dim):
    #         train_id = i % n_train
    #         train_d = i // n_train - 1
    #         K_Xf[i][j] = ff_kernel(X_train[train_id], x_test, lens, train_d, j)

    mu_frs = K_fX @ np.linalg.inv(noise_kernel) @ y_train
    var_frs = np.diag(K_test_test)[1:] - np.diag(K_fX @ np.linalg.inv(noise_kernel) @ K_fX.T)
    
    return mu_nrg, var_nrg, -1.0 * mu_frs.reshape(-1, 3), var_frs.reshape(-1, 3)


class GPR_Calc(Calculator):
    implemented_properties = ['energy', 'forces', 'energy_std', 'forces_std']

    def __init__(self, X_train, y_train, theta, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.X_train = X_train
        self.y_train = y_train
        self.theta = theta

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        x_test = self.atoms.positions.flatten()
        mu_nrg, var_nrg, mu_frs, var_frs = gpr_predict(self.X_train, self.y_train, x_test, self.theta)

        self.energy = mu_nrg
        self.uncertainty = var_nrg ** 0.5
        self.forces = mu_frs
        self.results['energy'] = mu_nrg
        self.results['energy_std'] = var_nrg ** 0.5  # use the key "free_energy"  to store uncertainty
        self.results['forces'] = mu_frs
        self.results['forces_std'] = var_frs ** 0.5  # N_atom * 3


class BFGS_GPR(BFGS):
	def __init__(self, atoms, restart=None, logfile='-', trajectory=None, maxstep=0.04, 
				master=None, alpha=None, frs_std=None):
		super().__init__(atoms, restart, logfile, trajectory, maxstep, master)
		self.frs_std = frs_std

	def converged(self, forces=None):
		if forces is None:
			forces = self.atoms.get_forces()
		if hasattr(self.atoms, "get_curvature"):
			return (forces ** 2).sum(axis=1).max() < self.fmax ** 2 \
				and self.atoms.get_curvature() < 0.0
		condition1 = (forces ** 2).sum(axis=1).max() < self.fmax ** 2
		condition2 = self.atoms.calc.results['forces_std'].max() > self.frs_std
		return condition1 or condition2

	def log(self, forces=None):
		if forces is None:
			forces = self.atoms.get_forces()
		fmax = sqrt((forces ** 2).sum(axis=1).max())
		e = self.atoms.get_potential_energy(
			force_consistent=self.force_consistent
		)

		nrg_std = round(self.atoms.calc.results['energy_std'], 4)
		T = time.localtime()
		if self.logfile is not None:
			name = self.__class__.__name__
			if self.nsteps == 0:
				args = (" " * len(name), "Step", "Time", "Energy", "fmax", "nrg_std")
				msg = "%s  %4s %8s %15s %12s %12s \n" % args
				self.logfile.write(msg)

				if self.force_consistent:
					msg = "*Force-consistent energies used in optimization.\n"
					self.logfile.write(msg)

			ast = {1: "*", 0: ""}[self.force_consistent]
			args = (name, self.nsteps, T[3], T[4], T[5], e, ast, fmax, nrg_std)
			msg = "%s:  %3d %02d:%02d:%02d %15.6f%1s %12.4f %12.4f \n" % args
			self.logfile.write(msg)

			self.logfile.flush()