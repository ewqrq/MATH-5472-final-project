import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm
import time

from mfmodel import MFModel, generate_mlr_model, generate_data, row_col_selections

# Set random seeds for reproducibility
np.random.seed(1001)

# Create directories for outputs
Path("plots").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# Setup experiment parameters
mtype = "medium_mlr_hier_L5"
n = 5000  # Number of features
signal_to_noise = 4.0  # Signal-to-noise ratio
L = 5  # Number of levels
ranks = np.array([10, 7, 5, 2, 1])  # Hierarchical ranks
rank = ranks.sum()
nsamples = 150  # Number of samples for good estimation

print(f"Running experiments with:")
print(f"- Features (n): {n}")
print(f"- Samples: {nsamples}")
print(f"- Total rank: {rank}")
print(f"- Signal-to-noise ratio: {signal_to_noise}")
print(f"- Hierarchical ranks: {ranks}")

# Create hierarchical partition
pi_rows = np.random.permutation(n)
pi_inv_rows = np.argsort(pi_rows)
hpart = {'rows': {'pi': pi_rows, 'lk': []}, 'cols': {'pi': pi_rows, 'lk': []}}

# Create level partitions matching the number of levels
level_groups = [2, 8, 32, 64, n]  # Geometric progression for better block sizes
for ngroups in level_groups:
    lk = np.linspace(0, n, ngroups, endpoint=True, dtype=int)
    hpart['rows']['lk'].append(lk)
    hpart['cols']['lk'].append(lk)

print(f"- Block sizes: {[len(lk)-1 for lk in hpart['rows']['lk']]}")

# Generate true model and data
print("\nGenerating true model and data...")
true_mlr, true_sparse_F, true_D_noise = generate_mlr_model(n, hpart, ranks, signal_to_noise)
row_selectors, si_groups, F_hpart = row_col_selections(hpart)

# Add inverse permutations
F_hpart["pi_inv"] = pi_inv_rows
hpart["rows"]["pi_inv"] = pi_inv_rows
hpart["cols"]["pi_inv"] = pi_inv_rows

# Generate data
print("Generating initial data...")
C = generate_data(true_sparse_F, true_D_noise, nsamples, true_mlr)  # n x nsamples
Z = (C - C.mean(axis=1, keepdims=True))[pi_rows, :]  # Apply permutation
unpermuted_A = (Z @ Z.T / (Z.shape[1]-1))[pi_inv_rows, :][:, pi_inv_rows]  # Unpermute for covariance

# Compute true model log-likelihood
print("Computing true model log-likelihood...")
true_F, true_D = true_mlr.F, true_D_noise
true_mfm = MFModel(F=true_F, D=true_D, hpart=F_hpart, ranks=ranks)
true_mfm.inv_coefficients(det=True)

true_train_obj = true_mfm.log_likelihood(Z.T)  # Use built-in method
exp_true_ll = true_mfm.fast_exp_true_loglikelihood_value()  # No need for args
print(f"True model: train_ll={true_train_obj:.4f}, exp_ll={exp_true_ll:.4f}")

# Run multiple trials
n_trials = 3  # Number of trials
results = {
    'frob': {'train': [], 'exp': [], 'convergence': [], 'times': []},
    'mle': {'train': [], 'exp': [], 'convergence': [], 'times': []}
}

for trial in range(n_trials):
    print(f"\nTrial {trial + 1}/{n_trials}")
    
    # Generate new data for this trial
    print("Generating new data...")
    C = generate_data(true_sparse_F, true_D_noise, nsamples, true_mlr)  # n x nsamples
    Z = (C - C.mean(axis=1, keepdims=True))[pi_rows, :]  # n x nsamples
    
    # 1. Frobenius norm fitting
    print("Fitting with Frobenius norm...")
    start_time = time.time()
    frob_model, frob_losses = true_mfm.fast_frob_fit_loglikehood(
        unpermuted_A, Z.T, F_hpart, hpart, ranks, 
        printing=True
    )
    frob_time = time.time() - start_time
    
    # Compute log-likelihoods using the same method for both models
    frob_model.inv_coefficients(det=True)  # Important: compute inverse coefficients first
    frob_train_ll = frob_model.log_likelihood(Z.T)
    frob_exp_ll = frob_model.fast_exp_loglikelihood_value(
        np.concatenate([true_F, np.sqrt(true_D).reshape(-1, 1)], axis=1),
        ranks, hpart["rows"], row_selectors, si_groups
    )
    results['frob']['train'].append(frob_train_ll)
    results['frob']['exp'].append(frob_exp_ll)
    results['frob']['convergence'].append(frob_losses)
    results['frob']['times'].append(frob_time)
    
    # 2. MLE fitting
    print("Fitting with MLE...")
    start_time = time.time()
    mle_model = MFModel(hpart=F_hpart, ranks=ranks)
    mle_converged, mle_losses = mle_model.fit(
        Z, 
        max_iter=150,
        tol=1e-4,
        verbose=True,
        return_losses=True
    )
    mle_time = time.time() - start_time
    
    # Compute log-likelihoods
    mle_model.inv_coefficients(det=True)  # Important: compute inverse coefficients first
    mle_train_ll = mle_model.log_likelihood(Z.T)
    mle_exp_ll = mle_model.fast_exp_loglikelihood_value(
        np.concatenate([true_F, np.sqrt(true_D).reshape(-1, 1)], axis=1),
        ranks, hpart["rows"], row_selectors, si_groups
    )
    results['mle']['train'].append(mle_train_ll)
    results['mle']['exp'].append(mle_exp_ll)
    results['mle']['convergence'].append(mle_losses)
    results['mle']['times'].append(mle_time)
    
    print(f"Frobenius: train_ll={frob_train_ll:.4f}, exp_ll={frob_exp_ll:.4f}, time={frob_time:.2f}s")
    print(f"MLE: train_ll={mle_train_ll:.4f}, exp_ll={mle_exp_ll:.4f}, time={mle_time:.2f}s")

# Convert to numpy arrays for analysis
for method in ['frob', 'mle']:
    for metric in ['train', 'exp', 'times']:
        results[method][metric] = np.array(results[method][metric])

# Compute statistics
diff = results['mle']['exp'] - results['frob']['exp']
mean_diff = np.mean(diff)
std_diff = np.std(diff)
percent_mle_better = np.sum(diff > 0) / len(diff) * 100

print("\nSummary Statistics:")
print(f"Mean difference (MLE - Frob): {mean_diff:.4f} ± {std_diff:.4f}")
print(f"MLE better in {percent_mle_better:.1f}% of trials")
print("\nTiming Statistics:")
print(f"Frobenius: {results['frob']['times'].mean():.2f}s ± {results['frob']['times'].std():.2f}s")
print(f"MLE: {results['mle']['times'].mean():.2f}s ± {results['mle']['times'].std():.2f}s")

# Plot convergence curves with better visualization
plt.figure(figsize=(12, 6))

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot raw convergence curves
for trial in range(n_trials):
    # Plot Frobenius norm convergence
    frob_curve = np.array(results['frob']['convergence'][trial])
    ax1.semilogy(range(len(frob_curve)), frob_curve, 'b-', alpha=0.3, label=f'Frobenius (Trial {trial+1})' if trial == 0 else None)
    
    # Plot MLE convergence
    mle_curve = np.array(results['mle']['convergence'][trial])
    ax1.semilogy(range(len(mle_curve)), -np.array(mle_curve), 'r-', alpha=0.3, label=f'MLE (Trial {trial+1})' if trial == 0 else None)

ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss (log scale)')
ax1.set_title('Raw Convergence Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot normalized convergence curves
for trial in range(n_trials):
    # Normalize Frobenius curve
    frob_curve = np.array(results['frob']['convergence'][trial])
    frob_norm = (frob_curve - frob_curve.min()) / (frob_curve.max() - frob_curve.min())
    ax2.plot(range(len(frob_curve)), frob_norm, 'b-', alpha=0.3, label=f'Frobenius (Trial {trial+1})' if trial == 0 else None)
    
    # Normalize MLE curve
    mle_curve = np.array(results['mle']['convergence'][trial])
    mle_norm = (mle_curve - mle_curve.min()) / (mle_curve.max() - mle_curve.min())
    ax2.plot(range(len(mle_curve)), mle_norm, 'r-', alpha=0.3, label=f'MLE (Trial {trial+1})' if trial == 0 else None)

ax2.set_xlabel('Iteration')
ax2.set_ylabel('Normalized Loss')
ax2.set_title('Normalized Convergence Curves')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"plots/convergence_{mtype}.pdf", bbox_inches='tight', dpi=300)
plt.close()

# Plot results with default style
plt.figure(figsize=(8, 5))
plt.hist(results['frob']['exp'], bins=15, alpha=0.5, color='blue', label='Frobenius', density=True)
plt.hist(results['mle']['exp'], bins=15, alpha=0.5, color='orange', label='MLE', density=True)

plt.axvline(np.mean(results['frob']['exp']), color='blue', linestyle='--', linewidth=1)
plt.axvline(np.mean(results['mle']['exp']), color='orange', linestyle='--', linewidth=1)

plt.xlabel('Expected Log-Likelihood')
plt.ylabel('Density')
plt.title('Distribution of Expected Log-Likelihood')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"plots/hist_{mtype}.pdf", bbox_inches='tight', dpi=300)
plt.close()

# Plot 2: Difference distribution
plt.figure(figsize=(8, 5))
plt.hist(diff, bins=15, alpha=0.5, color='green', density=True)
plt.axvline(mean_diff, color='darkgreen', linestyle='--', linewidth=1)

plt.xlabel('MLE - Frobenius Log-Likelihood Difference')
plt.ylabel('Density')
plt.title('Distribution of Log-Likelihood Differences')
plt.grid(True, alpha=0.3)
plt.savefig(f"plots/hist_diff_{mtype}.pdf", bbox_inches='tight', dpi=300)
plt.close()

# Save results
filename = f"{mtype}_r{rank}_{n}"
full_results = {
    'parameters': {
        'n': n,
        'nsamples': nsamples,
        'ranks': ranks,
        'signal_to_noise': signal_to_noise,
        'n_trials': n_trials,
        'block_sizes': [len(lk)-1 for lk in hpart['rows']['lk']]
    },
    'results': results,
    'statistics': {
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'percent_mle_better': percent_mle_better,
        'timing': {
            'frob_mean': float(results['frob']['times'].mean()),
            'frob_std': float(results['frob']['times'].std()),
            'mle_mean': float(results['mle']['times'].mean()),
            'mle_std': float(results['mle']['times'].std())
        }
    }
}

with open(f'outputs/hist_ll_{filename}.pickle', 'wb') as f:
    pickle.dump(full_results, f)

print("\nResults saved to outputs/hist_ll_{filename}.pickle")
print("Plots saved to:")
print(f"- plots/convergence_{mtype}.pdf")
print(f"- plots/hist_{mtype}.pdf")
print(f"- plots/hist_diff_{mtype}.pdf")
