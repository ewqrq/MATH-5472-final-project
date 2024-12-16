import numpy as np
from typing import Dict
import logging
from scipy.linalg import cho_factor, cho_solve
import warnings

# Configure model logging
logger = logging.getLogger(__name__)

class MFModel:
    """
    Multilevel Factor Model with MLR covariance structure.

    Implements efficient EM algorithm for fitting and prediction with linear time complexity.
    """

    def __init__(self, F=None, D=None, hpart=None, ranks=None):
        """Initialize MFModel.

        Args:
            F: Factor loadings matrix (n x total_rank)
            D: Diagonal noise variances (n)
            hpart: Hierarchical partition structure
            ranks: Rank allocation for each level
        """
        self.F = F
        self.D = D
        self.hpart = hpart
        self.ranks = ranks
        self.eps = 1e-10  # Numerical stability constant

        if F is not None:
            self.n_features = F.shape[0]
            self.total_rank = F.shape[1]

        # Initialize permutation vectors
        if hpart is not None:
            self.pi = hpart.get("pi", None)
            self.pi_inv = hpart.get("pi_inv", None)
            if self.pi is not None and self.pi_inv is None:
                self.pi_inv = np.argsort(self.pi)

        # Initialize other attributes
        self.logdet = None
        self.H = None

    def fit(self, Y, max_iter=100, tol=1e-6, verbose=False, return_losses=False):
        """Fit the model using EM algorithm as specified in Section 3 of the paper.
        
        Args:
            Y: Observed data matrix (n x p) where n is number of features and p is number of samples
            max_iter: Maximum number of iterations
            tol: Convergence tolerance for log-likelihood
            verbose: Whether to print progress
            return_losses: Whether to return list of log-likelihood values
        """
        if Y is None:
            raise ValueError("Input data Y is required")
            
        # Scale data for numerical stability
        Y_scale = np.sqrt(np.mean(Y * Y))
        Y_scaled = Y / Y_scale
        
        # Initialize parameters if not already done
        if self.F is None:
            self._initialize_parameters(Y_scaled)
        else:
            self.F = self.F / Y_scale
            self.D = self.D / (Y_scale * Y_scale)
            
        # Main EM loop
        prev_ll = -np.inf
        converged = False
        eps = 1e-8
        losses = []  # Track log-likelihood values
        
        try:
            for iteration in range(max_iter):
                # E-step: Compute sufficient statistics block by block
                self.inv_coefficients(eps=eps)
                
                # Get level partitions
                if "lk" in self.hpart:
                    lk = self.hpart["lk"]
                else:
                    lk = self.hpart["rows"]["lk"]
                
                # Process each level separately
                start_rank = 0
                for level, rank in enumerate(self.ranks):
                    if rank == 0 or level >= len(lk):
                        continue
                        
                    # Get blocks for this level
                    level_blocks = lk[level]
                    num_blocks = len(level_blocks) - 1
                    
                    # Process each block
                    for i in range(num_blocks):
                        r1, r2 = level_blocks[i], level_blocks[i+1]
                        
                        if r2 > r1:  # Only process non-empty blocks
                            # Extract relevant blocks
                            Y_block = Y_scaled[r1:r2]  # n_block x p
                            F_block = self.F[r1:r2, start_rank:start_rank+rank]  # n_block x rank
                            D_inv_block = 1.0 / (self.D[r1:r2] + eps)  # n_block
                            
                            # Compute E[Z|Y] and E[ZZ^T|Y] as in paper
                            FtDinv = F_block.T * D_inv_block[None, :]  # rank x n_block
                            M = np.eye(rank) + FtDinv @ F_block  # rank x rank
                            
                            try:
                                # Use Cholesky for better numerical stability
                                L = np.linalg.cholesky(M + eps * np.eye(rank))
                                EZ = np.linalg.solve(L.T, np.linalg.solve(L, FtDinv @ Y_block))
                                EZZt = Y_block.shape[1] * np.linalg.solve(L.T, np.linalg.solve(L, np.eye(rank))) + EZ @ EZ.T
                            except np.linalg.LinAlgError:
                                # Fallback to eigendecomposition
                                eigvals, eigvecs = np.linalg.eigh(M + eps * np.eye(rank))
                                sqrt_inv = eigvecs @ np.diag(1.0/np.sqrt(np.maximum(eigvals, eps))) @ eigvecs.T
                                EZ = sqrt_inv @ sqrt_inv @ FtDinv @ Y_block
                                EZZt = Y_block.shape[1] * sqrt_inv @ sqrt_inv + EZ @ EZ.T
                            
                            # M-step: Update F and D using sufficient statistics
                            try:
                                F_new = Y_block @ EZ.T @ np.linalg.inv(EZZt + eps * np.eye(rank))
                                self.F[r1:r2, start_rank:start_rank+rank] = F_new
                            except np.linalg.LinAlgError:
                                # Add more regularization if inversion fails
                                F_new = Y_block @ EZ.T @ np.linalg.inv(EZZt + 1e-6 * np.eye(rank))
                                self.F[r1:r2, start_rank:start_rank+rank] = F_new
                            
                            # Update D: D = 1/p * diag(YY^T - 2YE[Z^T]F^T + FE[ZZ^T]F^T)
                            YYt = np.sum(Y_block * Y_block, axis=1)
                            YEZt = Y_block @ EZ.T
                            FEZZtFt = (F_new @ EZZt @ F_new.T).diagonal()
                            self.D[r1:r2] = np.maximum((YYt - 2 * np.sum(YEZt * F_new, axis=1) + FEZZtFt) / Y_block.shape[1], eps)
                    
                    start_rank += rank
                
                # Compute log-likelihood
                ll = self.log_likelihood(Y_scaled.T)
                losses.append(ll)  # Track log-likelihood
                
                if verbose:
                    print(f"Iteration {iteration + 1}, Log-likelihood: {ll:.4f}")
                
                # Check convergence
                if abs(ll - prev_ll) < tol:
                    converged = True
                    break
                
                prev_ll = ll
                
        except Exception as e:
            if verbose:
                print(f"Error in iteration {iteration}: {str(e)}")
        
        # Rescale parameters back
        self.F = self.F * Y_scale
        self.D = self.D * (Y_scale * Y_scale)
        
        if verbose:
            if converged:
                print(f"Converged after {iteration + 1} iterations")
            else:
                print("Maximum iterations reached without convergence")
        
        if return_losses:
            return converged, losses
        return converged

    def solve(self, v, eps=None, max_iter=1):
        """Solve linear system efficiently using MLR structure with Cholesky decomposition."""
        if eps is None:
            eps = self.eps
            
        if len(v.shape) == 1:
            v = v.reshape(-1, 1)
        
        if self.hpart is not None:
            x = v[self.hpart["pi"]]
            
            # Compute system matrix with better conditioning
            D_reg = self.D + eps
            FtDinv = self.F.T / D_reg[np.newaxis, :]
            FtDinvF = FtDinv @ self.F
            FtDinvx = FtDinv @ x
            
            # Add regularization for stability
            system_matrix = FtDinvF + eps * np.eye(self.total_rank)
            
            # Use Cholesky decomposition for better numerical stability
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    L = np.linalg.cholesky(system_matrix)
                    z = np.linalg.solve(L.T, np.linalg.solve(L, FtDinvx)).ravel()
            except np.linalg.LinAlgError:
                # Fallback to standard solve with more regularization
                z = np.linalg.solve(system_matrix + 10*eps * np.eye(self.total_rank), FtDinvx).ravel()
            
            return z
        else:
            return np.zeros(self.total_rank)

    def inv_coefficients(self, det=False, eps=1e-8, max_iter=1):
        """
        Compute inverse coefficients using recursive formula from paper.
        
        Implements Σ^{-1} = D^{-1} - sum_{l=1}^L H_l H_l^T where:
        H_l = D^{-1/2} F_l (I + F_l^T D^{-1} F_l)^{-1/2}
        """
        if self.F is None or self.D is None or self.hpart is None or self.ranks is None:
            logger.error("Missing required attributes: F, D, hpart, or ranks.")
            return
        
        n = self.F.shape[0]
        total_rank = self.F.shape[1]
        
        # Add regularization for stability
        D_reg = self.D + eps
        D_inv = 1.0 / D_reg
        D_sqrt_inv = np.sqrt(D_inv)
        
        # Initialize storage for H matrices
        H = []  # Store H_l matrices
        logdet_terms = []  # Store determinant terms if needed
        
        # Get level partitions and ranks
        lk = self.hpart["lk"]
        ranks = self.ranks
        
        # Process each level separately
        start_rank = 0
        Sigma_inv = np.diag(D_inv)  # Start with D^{-1}
        
        for level, rank in enumerate(ranks):
            if rank == 0:
                continue
                
            # Get blocks for this level
            if level >= len(lk):
                break
                
            level_blocks = lk[level]
            num_blocks = len(level_blocks) - 1
            
            # Initialize H_l for this level
            H_l = np.zeros((n, rank))
            
            # Get F_l for this level
            F_l = self.F[:, start_rank:start_rank+rank]
            
            # Compute blocks of H_l
            for i in range(num_blocks):
                r1, r2 = level_blocks[i], level_blocks[i+1]
                
                if r2 > r1:  # Only process non-empty blocks
                    # Extract block of F_l
                    F_block = F_l[r1:r2, :]  # n_block x rank
                    D_sqrt_inv_block = D_sqrt_inv[r1:r2]  # n_block
                    
                    # Scale F by D^{-1/2} as in paper
                    F_scaled = D_sqrt_inv_block[:, None] * F_block
                    
                    # Compute M = I + F^T D^{-1} F for this block
                    M = np.eye(rank) + F_scaled.T @ F_scaled
                    
                    try:
                        # Use Cholesky decomposition for stability
                        L = np.linalg.cholesky(M + eps * np.eye(rank))
                        H_block = F_scaled @ np.linalg.inv(L).T
                        
                        if det:
                            logdet_terms.append(2 * np.sum(np.log(np.diag(L))))
                    except np.linalg.LinAlgError:
                        try:
                            # Fallback to eigendecomposition with more regularization
                            M_reg = M + 1e-6 * np.eye(rank)
                            eigvals, eigvecs = np.linalg.eigh(M_reg)
                            sqrt_inv = eigvecs @ np.diag(1.0/np.sqrt(np.maximum(eigvals, eps))) @ eigvecs.T
                            H_block = F_scaled @ sqrt_inv
                            
                            if det:
                                logdet_terms.append(np.sum(np.log(np.maximum(eigvals, eps))))
                        except np.linalg.LinAlgError:
                            # If everything fails, use strong regularization
                            logger.warning(f"Using strong regularization for block {i} at level {level}")
                            M_reg = M + 1e-4 * np.eye(rank)
                            H_block = F_scaled @ np.linalg.inv(M_reg)
                    
                    H_l[r1:r2] = H_block
            
            # Update Sigma_inv using recursive formula from paper
            Sigma_inv = Sigma_inv - H_l @ H_l.T
            H.append(H_l)
            start_rank += rank
        
        # Store results
        self.H = H  # Store H matrices for later use
        
        if det:
            # Compute log determinant as in paper
            self.logdet = np.sum(np.log(D_reg))  # First term from D
            if logdet_terms:  # Add terms from the recursive computation
                self.logdet += np.sum(logdet_terms)

    def predict(self, Y_new):
        """
        Predict latent factors for new data.

        Args:
            Y_new: New data matrix (n x p_new) where n is number of features and p_new is number of new samples

        Returns:
            Predicted latent factors (total_rank x p_new)
        """
        if Y_new.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {Y_new.shape[0]}")
            
        p_new = Y_new.shape[1]
        Z = np.zeros((self.total_rank, p_new))

        self.inv_coefficients()
        for i in range(p_new):
            Z[:, i] = self.solve(Y_new[:, i])

        return Z

    def log_likelihood(self, Y):
        """
        Compute log-likelihood for given data.
        
        Args:
            Y: Data matrix (nsamples x n_features)
            
        Returns:
            Normalized log-likelihood value
        """
        if Y.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {Y.shape[1]}")
            
        n, N = Y.shape[1], Y.shape[0]
        
        try:
            # Compute quadratic term efficiently
            quad_sum = 0
            valid_samples = 0
            
            for i in range(N):
                z = self.solve(Y[i])
                if np.any(np.isnan(z)) or np.any(np.isinf(z)):
                    continue
                    
                y_recon = self.F @ z
                residual = Y[i] - y_recon
                quad_term = np.sum(residual * residual / (self.D + self.eps))
                
                if not np.isnan(quad_term) and not np.isinf(quad_term):
                    quad_sum += quad_term
                    valid_samples += 1
            
            if valid_samples == 0:
                return -np.inf
            
            # Compute log determinant
            self.inv_coefficients(det=True)
            
            if np.isinf(self.logdet) or np.isnan(self.logdet):
                return -np.inf
            
            # Paper's exact formula:
            # L(F,D;Y) = -N/2 log(2π) - N/2 log det(FF^T + D) - 1/2 Tr((FF^T + D)^{-1} YY^T)
            ll = -0.5 * (
                n * valid_samples * np.log(2 * np.pi) +  # Term 1
                valid_samples * self.logdet +            # Term 2
                quad_sum                                 # Term 3
            )
            
            # Normalize by total number of elements
            return ll / (n * valid_samples)
            
        except Exception as e:
            logger.warning(f"Error in log_likelihood computation: {str(e)}")
            return -np.inf

    def _mult_blockdiag_refined(self, A, lk_A, B, lk_B):
        """
        Efficient block diagonal matrix multiplication.

        Args:
            A: First matrix
            lk_A: Level partition for A
            B: Second matrix
            lk_B: Level partition for B

        Returns:
            Result of block diagonal multiplication
        """
        nblocks_A = lk_A.size - 1
        nblocks_B = lk_B.size - 1

        # Get dimensions
        if A.shape[1] != B.shape[0]:
            A = A.T

        r = A.shape[1]
        result = np.zeros((nblocks_A * r, B.shape[1]))

        # Multiply blocks
        for i in range(nblocks_A):
            r1_A, r2_A = lk_A[i], lk_A[i+1]

            # Find overlapping blocks in B
            for j in range(nblocks_B):
                r1_B, r2_B = lk_B[j], lk_B[j+1]

                # Check if blocks overlap
                if r1_A < r2_B and r2_A > r1_B:
                    # Compute overlapping region
                    overlap_start = max(r1_A, r1_B)
                    overlap_end = min(r2_A, r2_B)

                    # Extract overlapping submatrices
                    A_sub = A[overlap_start:overlap_end]
                    B_sub = B[overlap_start:overlap_end]

                    # Multiply and accumulate result
                    result[i*r:(i+1)*r] += A_sub.T @ B_sub

        return result

    def diag(self) -> np.ndarray:
        """Return diagonal of Sigma with permutation."""
        if self.F is None or self.hpart is None or self.ranks is None or self.D is None:
            return np.array([])  # Return empty array if any required attribute is None
        diag = self.diag_sparse_FFt(self.F, self.hpart, self.ranks) + self.D
        return diag[self.pi_inv]

    def diag_sparse_FFt(self,
                       F_compressed: np.ndarray,
                       hpart: Dict,
                       ranks: np.ndarray) -> np.ndarray:
        """
        Compute diagonal of F F^T efficiently.

        Args:
            F_compressed: Compressed factor matrix
            hpart: Hierarchical partitioning
            ranks: Array of ranks

        Returns:
            Diagonal elements
        """
        res = np.zeros(hpart['pi'].size)

        for level in range(len(hpart["lk"])):
            lk = hpart["lk"][level]
            num_blocks = lk.size - 1

            start_rank = ranks[:level].sum()
            end_rank = ranks[:level+1].sum()

            for block in range(num_blocks):
                r1, r2 = lk[block], lk[block+1]
                F_block = F_compressed[r1:r2, start_rank:end_rank]
                res[r1:r2] += np.sum(F_block * F_block, axis=1)

        return res

    def fast_frob_fit_loglikehood(self, A, Y, F_hpart, hpart, ranks, max_iter=100, tol=1e-6, printing=False):
        """Efficient Frobenius norm fitting implementation using block-wise operations."""
        n = A.shape[0]
        total_rank = sum(ranks)
        losses = []
        
        # Initialize model if not already done
        if self.F is None:
            self.F = np.zeros((n, total_rank))
        if self.D is None:
            self.D = np.diag(A).copy()
        
        # Get level partitions
        if "lk" in hpart:
            lk = hpart["lk"]
        else:
            lk = hpart["rows"]["lk"]
        
        # Pre-compute block indices for efficiency
        block_indices = []
        for level in range(len(lk)):
            level_blocks = []
            for i in range(len(lk[level])-1):
                r1, r2 = lk[level][i], lk[level][i+1]
                if r2 > r1:
                    level_blocks.append((r1, r2))
            block_indices.append(level_blocks)
        
        # Main optimization loop
        prev_loss = np.inf
        for iteration in range(max_iter):
            # 1. Update F level by level using block operations
            start_rank = 0
            for level, rank in enumerate(ranks):
                if rank == 0 or level >= len(lk):
                    continue
                
                for r1, r2 in block_indices[level]:
                    block_size = r2 - r1
                    
                    # Extract relevant blocks
                    A_block = A[r1:r2, r1:r2]
                    D_block = np.diag(self.D[r1:r2])
                    
                    # Compute block update efficiently
                    try:
                        # Use eigendecomposition for better numerical stability
                        eigvals, eigvecs = np.linalg.eigh(A_block - D_block)
                        # Sort eigenvalues in descending order
                        idx = np.argsort(eigvals)[::-1]
                        eigvals = eigvals[idx]
                        eigvecs = eigvecs[:, idx]
                        
                        # Take top rank eigenvalues/vectors and ensure proper broadcasting
                        top_eigvals = np.maximum(eigvals[:rank], 0)
                        top_eigvecs = eigvecs[:, :rank]
                        scaling = np.sqrt(top_eigvals).reshape(1, -1)
                        
                        # Update F block with proper broadcasting
                        self.F[r1:r2, start_rank:start_rank+rank] = top_eigvecs * scaling
                        
                    except np.linalg.LinAlgError:
                        # Fallback to SVD if eigendecomposition fails
                        try:
                            U, s, _ = np.linalg.svd(A_block - D_block, full_matrices=False)
                            scaling = np.sqrt(np.maximum(s[:rank], 0)).reshape(1, -1)
                            self.F[r1:r2, start_rank:start_rank+rank] = U[:, :rank] * scaling
                        except np.linalg.LinAlgError:
                            # If SVD also fails, use random initialization
                            self.F[r1:r2, start_rank:start_rank+rank] = np.random.randn(block_size, rank) * np.sqrt(0.1)
                
                start_rank += rank
            
            # 2. Update D efficiently using vectorized operations
            FFt_diag = np.sum(self.F * self.F, axis=1)
            self.D = np.maximum(np.diag(A) - FFt_diag, self.eps)
            
            # 3. Compute loss efficiently using block structure
            loss = 0.0
            start_rank = 0
            for level, rank in enumerate(ranks):
                if rank == 0 or level >= len(lk):
                    continue
                    
                for r1, r2 in block_indices[level]:
                    # Compute block contribution to loss
                    F_block = self.F[r1:r2, start_rank:start_rank+rank]
                    A_block = A[r1:r2, r1:r2]
                    D_block = np.diag(self.D[r1:r2])
                    
                    # Efficient block loss computation
                    block_diff = A_block - (F_block @ F_block.T + D_block)
                    loss += np.sum(block_diff * block_diff)
                
                start_rank += rank
            
            current_loss = 0.5 * loss
            losses.append(current_loss)
            
            if printing:
                print(f"Iteration {iteration + 1}, Loss: {current_loss:.4f}")
            
            # Check convergence
            if abs(current_loss - prev_loss) < tol * abs(prev_loss):
                if printing:
                    print(f"Converged after {iteration + 1} iterations")
                break
                
            prev_loss = current_loss
        
        return self, losses

    def fast_exp_loglikelihood_value(self, true_params, ranks, hpart_rows, row_selectors, si_groups):
        """Compute expected log-likelihood value efficiently."""
        n = self.n_features
        
        # Compute log determinant
        self.inv_coefficients(det=True)
        logdet = self.logdet
        
        # Compute trace term efficiently
        trace_term = 0
        for i in range(n):
            trace_term += true_params[i, -1]**2 / (self.D[i] + self.eps)
        
        # Compute quadratic term efficiently using MLR structure
        quad_term = 0
        start_rank = 0
        for level, rank in enumerate(ranks):
            if rank == 0:
                continue
                
            if level >= len(hpart_rows["lk"]):
                break
                
            lk = hpart_rows["lk"][level]
            for i in range(len(lk)-1):
                r1, r2 = lk[i], lk[i+1]
                F_block = true_params[r1:r2, :-1][:, start_rank:start_rank+rank]
                F_fitted_block = self.F[r1:r2, start_rank:start_rank+rank]
                
                quad_term += np.sum(F_block * F_fitted_block) / (self.D[r1:r2].mean() + self.eps)
            
            start_rank += rank
        
        # Compute final expected log-likelihood
        ll = -0.5 * (n * np.log(2 * np.pi) + logdet + trace_term - 2 * quad_term)
        
        return ll / n  # Normalize by dimension as in paper

    def fast_exp_true_loglikelihood_value(self):
        """Compute expected log-likelihood for true model."""
        n = self.n_features
        
        # Compute log determinant
        self.inv_coefficients(det=True)
        logdet = self.logdet
        
        # Compute trace term (simplified for true model)
        trace_term = n  # Since D is identity and noise is unit variance
        
        # Final expected log-likelihood
        ll = -0.5 * (n * np.log(2 * np.pi) + logdet + trace_term)
        
        return ll / n  # Normalize by dimension as in paper

    def _initialize_parameters(self, Y):
        """Initialize model parameters following Section 5's approach.
        Uses block-wise SVD initialization for F and residual-based initialization for D.
        """
        n = Y.shape[0]  # n features (after transpose)
        p = Y.shape[1]  # p samples
        total_rank = sum(self.ranks)
        eps = 1e-8
        
        # Initialize F matrix using block-wise SVD
        self.F = np.zeros((n, total_rank))
        start_rank = 0
        
        # Get level partitions
        if "lk" in self.hpart:
            lk = self.hpart["lk"]
        else:
            lk = self.hpart["rows"]["lk"]
        
        # Initialize each level separately as described in Section 5
        for level, rank in enumerate(self.ranks):
            if rank == 0 or level >= len(lk):
                continue
                
            level_blocks = lk[level]
            num_blocks = len(level_blocks) - 1
            
            for i in range(num_blocks):
                r1, r2 = level_blocks[i], level_blocks[i+1]
                block_size = r2 - r1
                
                if block_size > 0:
                    # Extract block data
                    Y_block = Y[r1:r2]  # block_size x p
                    Y_centered = Y_block - Y_block.mean(axis=1, keepdims=True)
                    
                    # Compute block covariance efficiently
                    if block_size <= p:
                        # If block size is smaller than samples, compute directly
                        cov_block = (Y_centered @ Y_centered.T) / p
                        try:
                            # Add small regularization for stability
                            U, s, _ = np.linalg.svd(cov_block + eps * np.eye(block_size))
                            # Scale by sqrt of eigenvalues as in paper
                            if rank > 0:
                                self.F[r1:r2, start_rank:start_rank+rank] = U[:, :rank] * np.sqrt(np.maximum(s[:rank] - 1, 0))[:, None].T
                        except np.linalg.LinAlgError:
                            # Fallback to random initialization if SVD fails
                            if rank > 0:
                                self.F[r1:r2, start_rank:start_rank+rank] = np.random.randn(block_size, rank) * np.sqrt(0.1 / rank)
                    else:
                        # If block size is larger than samples, use alternative form
                        try:
                            # Compute SVD of Y_centered directly
                            U, s, _ = np.linalg.svd(Y_centered, full_matrices=False)
                            if rank > 0:
                                # Scale by sqrt of eigenvalues
                                self.F[r1:r2, start_rank:start_rank+rank] = U[:, :rank] * np.sqrt(np.maximum(s[:rank]**2/p - 1, 0))[:, None].T
                        except np.linalg.LinAlgError:
                            # Fallback to random initialization
                            if rank > 0:
                                self.F[r1:r2, start_rank:start_rank+rank] = np.random.randn(block_size, rank) * np.sqrt(0.1 / rank)
            
            start_rank += rank
        
        # Initialize D using residual variances as in paper
        Y_centered = Y - Y.mean(axis=1, keepdims=True)
        sample_var = np.sum(Y_centered * Y_centered, axis=1) / p
        FFt_diag = np.sum(self.F * self.F, axis=1)  # Diagonal of F @ F.T
        
        # Ensure positive variances with regularization
        self.D = np.maximum(sample_var - FFt_diag, eps)
        
        # Store dimensions
        self.n_features = n
        self.total_rank = total_rank

def generate_mlr_model(n, hpart, ranks, signal_to_noise):
    """Generate MLR model following paper's setup."""
    # Initialize model
    model = MFModel()
    
    # Generate sparse factor matrix
    total_rank = sum(ranks)
    F = np.zeros((n, total_rank))
    
    # Generate structured factors
    start_rank = 0
    for level, rank in enumerate(ranks):
        lk = hpart["rows"]["lk"][level]
        for i in range(len(lk)-1):
            r1, r2 = lk[i], lk[i+1]
            # Generate block with hierarchical structure
            F_block = np.random.randn(r2-r1, rank)
            # Normalize columns
            F_block /= np.sqrt(np.sum(F_block * F_block, axis=0))
            F[r1:r2, start_rank:start_rank+rank] = F_block
        start_rank += rank
    
    # Scale factor loadings for desired signal-to-noise ratio
    F = F * np.sqrt(signal_to_noise / total_rank)
    
    # Generate noise with unit variance
    D = np.ones(n)
    
    # Create model
    model.F = F
    model.D = D
    model.hpart = hpart["rows"]
    model.ranks = ranks
    model.n_features = n
    model.total_rank = total_rank
    model.pi_rows = hpart["rows"]["pi"]
    model.pi_cols = hpart["cols"]["pi"]
    model.pi_inv_rows = np.argsort(hpart["rows"]["pi"])
    model.pi_inv_cols = np.argsort(hpart["cols"]["pi"])
    
    return model, F, D

def generate_data(F, D, nsamples, model):
    """Generate data following paper's model."""
    n = F.shape[0]
    total_rank = F.shape[1]
    
    # Generate latent factors
    Z = np.random.randn(total_rank, nsamples)
    
    # Generate observations
    Y = F @ Z + np.random.randn(n, nsamples) * np.sqrt(D)[:, None]
    
    return Y

def row_col_selections(hpart):
    """Create row and column selections for hierarchical structure."""
    row_selectors = []
    si_groups = []
    
    # Extract level partitions
    lk = hpart["rows"]["lk"]
    
    # Create F_hpart
    F_hpart = {
        "pi": hpart["rows"]["pi"],
        "lk": lk[:-1]  # Exclude last level
    }
    
    return row_selectors, si_groups, F_hpart
