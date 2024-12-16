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

class ExperimentConfig:
    """Configuration class for experiments from the paper."""
    
    @staticmethod
    def asset_covariance():
        """Asset covariance matrix experiment (Section 5.1)"""
        return {
            'name': 'assetcov_gics',
            'n_features': 5000,
            'n_samples': 300,  # 300 trading days ending 2022/12/30
            'n_levels': 6,  # GICS hierarchy
            'ranks': {
                'fm': [29, 0, 0, 0, 0, 1],  # Traditional factor model
                'mfm': [14, 6, 4, 3, 2, 1]  # Multilevel factor model
            }
        }
    
    @staticmethod
    def synthetic_mlr():
        """Synthetic multilevel factor model experiment (Section 5.2)"""
        return {
            'name': 'medium_mlr_hier_L6',
            'n_features': 10000,
            'n_samples': 100,  # 4 * rank where rank = 25
            'n_levels': 6,
            'ranks': [10, 5, 4, 3, 2, 1],  # Total rank = 25
            'signal_to_noise': 4.0,
            'n_trials': 200  # For generating histograms
        }

def run_experiment(config):
    """Run experiment with given configuration."""
    print(f"\nRunning {config['name']} experiment:")
    print(f"- Features (n): {config['n_features']}")
    print(f"- Samples: {config['n_samples']}")
    print(f"- Levels: {config['n_levels']}")
    
    # Create hierarchical partition
    n = config['n_features']
    if config['name'] == 'synthetic_mlr':
        # Create partition as in merge_ex.py
        pi_rows = np.random.permutation(n)
        hpart = {'rows': {'pi': pi_rows, 'lk': []}, 'cols': {'pi': pi_rows, 'lk': []}}
        
        # Create level partitions matching paper's specification
        for ngroups in [2, 5, 9, 17, 33, n+1]:
            lk = np.linspace(0, n, ngroups, endpoint=True, dtype=int)
            hpart['rows']['lk'].append(lk)
            hpart['cols']['lk'].append(lk)
            
        ranks = config['ranks']
        signal_to_noise = config['signal_to_noise']
        
    else:  # Asset covariance case
        # TODO: Load GICS hierarchy from CRSP data
        # For now use placeholder similar structure
        pi_rows = np.random.permutation(n)
        hpart = {'rows': {'pi': pi_rows, 'lk': []}, 'cols': {'pi': pi_rows, 'lk': []}}
        
        # Approximate GICS-like hierarchy
        level_groups = [1, 11, 24, 69, 158, n]
        for ngroups in level_groups:
            lk = np.linspace(0, n, ngroups, endpoint=True, dtype=int)
            hpart['rows']['lk'].append(lk)
            hpart['cols']['lk'].append(lk)
            
        ranks = config['ranks']['mfm']  # Use MFM ranks by default
        signal_to_noise = 4.0  # Default SNR
    
    print(f"- Block sizes: {[len(lk)-1 for lk in hpart['rows']['lk']]}")
    print(f"- Ranks: {ranks}")
    
    # Generate true model and data
    print("\nGenerating true model and data...")
    true_mlr, true_sparse_F, true_D_noise = generate_mlr_model(n, hpart, ranks, signal_to_noise)
    row_selectors, si_groups, F_hpart = row_col_selections(hpart)
    
    # Add inverse permutations
    F_hpart["pi_inv"] = np.arange(n)
    F_hpart["pi"] = np.arange(n)
    permuted_F_hpart = {"pi_inv": np.arange(n), "pi": np.arange(n), "lk": F_hpart["lk"]}
    
    # Generate initial data
    print("Generating initial data...")
    C = generate_data(true_sparse_F, true_D_noise, config['n_samples'], true_mlr)
    Z = (C - C.mean(axis=1, keepdims=True))[hpart["rows"]["pi"], :]
    del C
    unpermuted_A = (Z @ Z.T / (Z.shape[1]-1))[true_mlr.pi_inv_rows, :][:, true_mlr.pi_inv_cols]
    Y = Z.T
    N = Y.shape[0]
    
    # Compute true model log-likelihood
    print("Computing true model log-likelihood...")
    true_F, true_D = true_mlr.B[:, :-1]+0, true_D_noise+0
    true_mfm = MFModel(F=true_F, D=true_D, hpart=F_hpart, ranks=ranks)
    true_mfm.inv_coefficients(det=True)
    
    true_train_obj = true_mfm.log_likelihood(Y[:, true_mfm.pi_inv])
    exp_true_ll = true_mfm.fast_exp_true_loglikelihood_value()
    print(f"TR: train ll={true_train_obj:.4f}, exp ll={exp_true_ll:.4f}")
    
    # Run trials
    n_trials = config.get('n_trials', 200)  # Default to 200 trials as in merge_ex.py
    results = {
        'frob': {'train': [], 'exp': [], 'convergence': [], 'times': []},
        'mle': {'train': [], 'exp': [], 'convergence': [], 'times': []}
    }
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Generate new data for this trial
        print("Generating new data...")
        C = generate_data(true_sparse_F, true_D_noise, config['n_samples'], true_mlr)
        Z = (C - C.mean(axis=1, keepdims=True))[hpart["rows"]["pi"], :]
        del C
        unpermuted_A = (Z @ Z.T / (Z.shape[1]-1))[true_mlr.pi_inv_rows, :][:, true_mlr.pi_inv_cols]
        Y = Z.T
        
        # 1. Frobenius norm fitting
        print("Fitting with Frobenius norm...")
        start_time = time.time()
        frob_model, frob_losses = true_mfm.fast_frob_fit_loglikehood(
            unpermuted_A, Y, F_hpart, hpart, ranks, 
            printing=False, eps_ff=1e-3
        )
        frob_time = time.time() - start_time
        
        # Compute log-likelihoods
        frob_model.inv_coefficients()
        frob_train_ll = frob_model.log_likelihood(Y[:, frob_model.pi_inv])
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
        mle_model, loglikelihoods = true_mfm.fit(
            Y, ranks, F_hpart,
            printing=False,
            max_iter=300,
            freq=100
        )
        mle_time = time.time() - start_time
        
        # Compute log-likelihoods
        mle_model.inv_coefficients()
        mle_train_ll = loglikelihoods[-1]
        mle_exp_ll = mle_model.fast_exp_loglikelihood_value(
            np.concatenate([true_F, np.sqrt(true_D).reshape(-1, 1)], axis=1),
            ranks, hpart["rows"], row_selectors, si_groups
        )
        results['mle']['train'].append(mle_train_ll)
        results['mle']['exp'].append(mle_exp_ll)
        results['mle']['convergence'].append(loglikelihoods)
        results['mle']['times'].append(mle_time)
        
        print(f"FR: train ll={frob_train_ll:.4f}, exp ll={frob_exp_ll:.4f}, time={frob_time:.2f}s")
        print(f"ML: train ll={mle_train_ll:.4f}, exp ll={mle_exp_ll:.4f}, time={mle_time:.2f}s")
    
    # Convert to numpy arrays for analysis
    for method in ['frob', 'mle']:
        for metric in ['train', 'exp', 'times']:
            results[method][metric] = np.array(results[method][metric])
    
    # Generate plots and save results
    plot_results(results, config)
    
    return results

def plot_results(results, config):
    """Generate plots exactly matching merge_ex.py format."""
    mtype = config['name']
    n = config['n_features']
    rank = sum(config['ranks'])
    filename = f"{mtype}_r{rank}_{n}"
    
    # Setup LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })
    
    # 1. Distribution of expected log-likelihoods
    fig, axs = plt.subplots(1, 1, figsize=(5, 3), dpi=180, sharey=True)
    
    mean1 = np.mean(results['frob']['exp'])
    std1 = np.std(results['frob']['exp'])
    mean2 = np.mean(results['mle']['exp'])
    std2 = np.std(results['mle']['exp'])
    
    print(len(results['frob']['exp']), "samples in histogram")
    
    plt.hist(results['frob']['exp'], bins=30, alpha=0.5, color='blue', 
             label='Frob', density=True, edgecolor='black')
    plt.hist(results['mle']['exp'], bins=22, alpha=0.5, color='yellow', 
             label='MLE', density=True, edgecolor='black')
    
    plt.axvline(mean1, color='darkblue', linestyle='--', linewidth=1)
    plt.axvline(mean2, color='darkgoldenrod', linestyle='--', linewidth=1)
    
    plt.xlabel(r'$\mathbf{E} (\ell(F, D; y))$')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(color='silver', linestyle='-', linewidth=0.3)
    axs.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/hist_{filename}.pdf", bbox_inches='tight')
    plt.close()
    
    # 2. Distribution of differences
    fig, axs = plt.subplots(1, 1, figsize=(5, 3), dpi=180, sharey=True)
    
    diff = np.array(results['mle']['exp']) - np.array(results['frob']['exp'])
    mean = np.mean(diff)
    
    print(len(results['frob']['exp']), "samples in histogram")
    plt.hist(diff, bins=30, alpha=0.5, color='green', label='MLE', 
             density=True, edgecolor='black')
    
    plt.axvline(mean, color='darkgreen', linestyle='--', linewidth=1)
    
    plt.xlabel(r'$\mathbf{E} (\ell(F^{\text{MLE}}, D^{\text{MLE}}; y))- \mathbf{E} (\ell(F^{\text{Frob}}, D^{\text{Frob}}; y))$')
    plt.ylabel('Density')
    plt.grid(color='silver', linestyle='-', linewidth=0.3)
    axs.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/hist_diff_{filename}.pdf", bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"Percentage where MLE worse than Frob: {diff[diff<0].size / diff.size * 100:.2f}%")
    print(f"Mean difference: {np.mean(diff):.4f}, Std: {np.std(diff):.4f}")
    
    # 3. Convergence plot for a single trial
    if len(results['mle']['convergence']) > 0:
        fig, axs = plt.subplots(1, 1, figsize=(5, 3), dpi=180, sharey=True)
        
        # Get last trial's convergence
        ll = np.array(results['mle']['convergence'][-1][1:])
        obj_frob = results['frob']['train'][-1]
        
        axs.plot(ll, color='r', lw=1, label=r"$\text{MLE}$")
        axs.set_xlabel('iteration')
        axs.axhline(obj_frob, 0, ll.size-1, color='b', lw=1, label=r"$\text{Frob}$")
        axs.set_ylabel(r'$\ell(F, D; Y)/N$')
        axs.set_ylim([obj_frob - 100, ll.max() + 10])
        axs.grid(True)
        axs.legend()
        
        plt.tight_layout()
        plt.savefig(f"plots/em_{filename}.pdf", bbox_inches='tight')
        plt.close()
        
        # Print final values
        print(f"FR: train ll={obj_frob:.4f}, exp ll={results['frob']['exp'][-1]:.4f}")
        print(f"ML: train ll={ll[-1]:.4f}, exp ll={results['mle']['exp'][-1]:.4f}")
        
        # Print differences
        print(f"Train diff: {np.round(ll[-1]) - np.round(obj_frob)}")
        print(f"Exp diff: {np.round(results['mle']['exp'][-1]) - np.round(results['frob']['exp'][-1])}")
    
    # Save results
    results_dict = {
        "frob": {"train": results['frob']['train'], "exp": results['frob']['exp']},
        "mle": {"train": results['mle']['train'], "exp": results['mle']['exp']}
    }
    
    with open(f"outputs/em_{filename}.pickle", 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # Run synthetic experiment
    synthetic_config = ExperimentConfig.synthetic_mlr()
    synthetic_results = run_experiment(synthetic_config)
    
    # Run asset covariance experiment
    asset_config = ExperimentConfig.asset_covariance()
    asset_results = run_experiment(asset_config)
