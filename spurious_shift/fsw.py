"""
fsw.py — FSW: Fair Class-Incremental Learning using Sample Weighting
         (Park, Kim & Whang, KDD 2026; reference code github.com/jyparkkr/FSW).

Domain-incremental adaptation of the reference EO variant (binary sensitive
attribute) for the spurious-shift benchmark; the class-incremental version
lives in bias_in_bios/fsw.py, with the full derivation in its docstring.

FSW assumes class-incremental tasks (new classes arrive, old ones live only in
the buffer). Here the label space is fixed, so the adaptation is: per-(y, z)
group statistics are computed over buffer ∪ current-task data jointly — in
class-IL that reduces exactly to the reference split (old classes from memory,
new from task data) because the class sets are disjoint — and both classes
count as "current" for the lambda accuracy regularizer (there are no
past-only classes, so lambda_old never applies). Sample weights are still
solved per epoch by the reference LP

    min_w  sum_y |lhat(G_{y,0}) - lhat(G_{y,1})|/2 + lambda * mean_y lhat(G_y)
    s.t.   w in [0,1]^m,  approximated class losses stay >= 0,

with lhat(G) = l(G) - alpha * < g_G_hat, sum_i w_i g_i_hat > the first-order
last-layer approximation, all gradients L2-normalized, and the learning rate
absorbed into alpha. Weights below FSW_DROP_THRESHOLD drop the sample for the
epoch; the first task trains plainly. Replay follows the paper's update
g <- g_curr + tau * g_mem with the same memory budget and batch size as the
other replay methods.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog
from torch.utils.data import DataLoader, TensorDataset

from model import (MLP, make_optimizer, make_scheduler, has_converged,
                   predict, batchnorm_eval)
from memory import EpisodicMemory
from config import (MLP_EPOCHS, MLP_BATCH_SIZE, MLP_GRAD_CLIP, DEVICE, SEED,
                    MEMORY_SIZE_PER_TASK, AGEM_MEM_BATCH_SIZE,
                    FSW_ALPHA, FSW_LAMBDA, FSW_TAU, FSW_DROP_THRESHOLD)


class FSW:
    """Fairness-aware Sample Weighting (EO objective, |Z| = 2, domain-IL)."""

    def __init__(self, input_dim, device=DEVICE, alpha=FSW_ALPHA, lmbd=FSW_LAMBDA,
                 tau=FSW_TAU, mem_size=MEMORY_SIZE_PER_TASK,
                 mem_batch=AGEM_MEM_BATCH_SIZE):
        self.input_dim = input_dim
        self.device = device
        self.alpha = alpha
        self.lmbd = lmbd
        self.tau = tau
        self.mem_batch_size = int(mem_batch)
        self.model = MLP(input_dim).to(device)
        self.optimizer = make_optimizer(self.model)
        self.memory = EpisodicMemory(size_per_task=int(mem_size), device=device)
        self._pad_rng = np.random.default_rng(SEED)
        self.weight_stats = {}   # task_idx -> per-epoch weight diagnostics

    def remember(self, X, y, s):
        self.memory.add_task(X, y, s)

    # -- last-layer closed-form stats ----------------------------------------

    @torch.no_grad()
    def _stats(self, X_t, y_t, batch_size=1024):
        """Per-sample loss, b_i = p_i - onehot(y_i), penultimate h_i (eval mode)."""
        was_training = self.model.training
        self.model.eval()
        losses, Bs, Hs = [], [], []
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(self.device)
            yb = y_t[i:i + batch_size].to(self.device)
            h = self.model.net[:-1](xb)
            logits = self.model.net[-1](h)
            losses.append(F.cross_entropy(logits, yb, reduction="none").cpu())
            p = F.softmax(logits, dim=1)
            Bs.append((p - F.one_hot(yb, num_classes=logits.shape[1]).to(p.dtype)).cpu())
            Hs.append(h.cpu())
        self.model.train(was_training)
        return torch.cat(losses), torch.cat(Bs), torch.cat(Hs)

    # -- LP construction ------------------------------------------------------

    def _compute_weights(self, X_t, y_t, s_t):
        if self.alpha == 0:   # reference early-exit: no gradient term
            return np.ones(len(X_t), dtype=np.float32)
        X_m, y_m, s_m = self.memory.get_all()
        loss_m, B_m, H_m = self._stats(X_m.cpu(), y_m.cpu())
        loss_c, B_c, H_c = self._stats(X_t, y_t)

        y_all = torch.cat([y_m.cpu(), y_t])
        s_all = torch.cat([s_m.cpu(), s_t])
        loss_all = torch.cat([loss_m, loss_c])
        B_all, H_all = torch.cat([B_m, B_c]), torch.cat([H_m, H_c])

        groups = {}
        for c in sorted(torch.unique(y_all).tolist()):
            for z in (0, 1):
                m = (y_all == c) & (s_all == z)
                ng = int(m.sum())
                if ng == 0:
                    continue
                Bg, Hg = B_all[m], H_all[m]
                groups[(c, z)] = (loss_all[m].mean().item(), Bg.mean(0), Bg.T @ Hg / ng)

        ordered = [c for c in sorted(torch.unique(y_all).tolist())
                   if (c, 0) in groups and (c, 1) in groups]
        n = len(ordered)
        if n == 0:
            return np.ones(len(X_t), dtype=np.float32)

        g_norm = (B_c.norm(dim=1) * (1.0 + H_c.pow(2).sum(1)).sqrt()).clamp_min(1e-12)

        def dots(beta, W):
            return (B_c @ beta + ((H_c @ W.T) * B_c).sum(1)) / g_norm

        A_rows, b_vec, C_rows, d_vec = [], [], [], []
        for c in ordered:
            l0, beta0, W0 = groups[(c, 0)]
            l1, beta1, W1 = groups[(c, 1)]
            nrm0 = torch.sqrt(beta0.pow(2).sum() + W0.pow(2).sum()).clamp_min(1e-12)
            nrm1 = torch.sqrt(beta1.pow(2).sum() + W1.pow(2).sum()).clamp_min(1e-12)
            beta0, W0 = beta0 / nrm0, W0 / nrm0
            beta1, W1 = beta1 / nrm1, W1 / nrm1
            b_vec.append((l0 - l1) / 2.0)
            A_rows.append(self.alpha * (dots(beta0, W0) - dots(beta1, W1)) / 2.0)
            d_vec.append((l0 + l1) / 2.0)
            C_rows.append(self.alpha * (dots(beta0, W0) + dots(beta1, W1)) / 2.0)

        A = (torch.stack(A_rows) / n).double().numpy()
        b = np.asarray(b_vec, dtype=np.float64) / n
        # all classes are current in domain-IL: lambda / n_cur, lambda_old unused
        C = (self.lmbd / n) * torch.stack(C_rows).double().numpy()
        d = (self.lmbd / n) * np.asarray(d_vec, dtype=np.float64)
        return self._solve_lp(A, b, C, d)

    @staticmethod
    def _solve_lp(A, b, C, d):
        """min_x sum|Ax - b| + sum(d - Cx)  s.t.  0<=x<=1, Cx <= d (reference LP)."""
        K, m = A.shape
        cost = np.concatenate([-C.sum(0), np.ones(K)])
        A_ub = sp.vstack([
            sp.hstack([sp.csr_matrix(A),  -sp.eye(K)]),
            sp.hstack([sp.csr_matrix(-A), -sp.eye(K)]),
            sp.hstack([sp.csr_matrix(C),  sp.csr_matrix((K, K))]),
        ], format="csr")
        b_ub = np.concatenate([b, -b, d])
        res = linprog(cost, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(0.0, 1.0)] * m + [(0.0, None)] * K,
                      method="highs")
        if not res.success:
            print(f"    [fsw] LP failed ({res.message}); using uniform weights")
            return np.ones(m, dtype=np.float32)
        return res.x[:m].astype(np.float32)

    # -- training --------------------------------------------------------------

    def train_task(self, X_train, y_train, s_train=None, epochs=MLP_EPOCHS):
        if s_train is None:
            s_train = np.zeros(len(y_train), dtype=np.int64)
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        s_t = torch.tensor(s_train, dtype=torch.long)
        task_idx = self.memory.num_tasks()
        criterion = nn.CrossEntropyLoss(reduction="none")
        scheduler = make_scheduler(self.optimizer, epochs)
        self.weight_stats[task_idx] = []

        epoch_losses = []
        for _ in range(epochs):
            drop_last = False
            if task_idx > 0:
                w = self._compute_weights(X_t, y_t, s_t)
                keep = w > FSW_DROP_THRESHOLD
                self.weight_stats[task_idx].append({
                    "kept": int(keep.sum()), "total": len(w),
                    "mean_w_s0": float(w[s_train == 0].mean()) if (s_train == 0).any() else float("nan"),
                    "mean_w_s1": float(w[s_train == 1].mean()) if (s_train == 1).any() else float("nan"),
                })
                # Pad the kept set to batch divisibility with below-threshold
                # samples (reference trick): their near-zero weights keep them
                # out of the loss; they only stabilise BatchNorm. Fall back to
                # drop_last if the pool can't prevent a size-1 final batch.
                idx = np.where(keep)[0]
                pad = (-len(idx)) % MLP_BATCH_SIZE
                if pad:
                    pool = np.where(~keep)[0]
                    take = min(pad, len(pool))
                    if take:
                        idx = np.concatenate([idx, self._pad_rng.choice(pool, take, replace=False)])
                drop_last = len(idx) % MLP_BATCH_SIZE == 1 and len(idx) > MLP_BATCH_SIZE
                idx_t = torch.from_numpy(idx)
                ds = TensorDataset(X_t[idx_t], y_t[idx_t],
                                   torch.tensor(w, dtype=torch.float32)[idx_t])
            else:
                ds = TensorDataset(X_t, y_t, torch.ones(len(X_t)))
            loader = DataLoader(ds, batch_size=MLP_BATCH_SIZE, shuffle=True,
                                drop_last=drop_last)

            self.model.train()
            batch_loss = 0.0
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(self.device), yb.to(self.device), wb.to(self.device)
                self.optimizer.zero_grad()
                loss = (criterion(self.model(xb), yb) * wb).mean()
                if task_idx > 0 and self.tau > 0:
                    mem = self.memory.get_random_batch(self.mem_batch_size)
                    if mem is not None:
                        X_mem, y_mem, _ = mem
                        with batchnorm_eval(self.model):
                            loss = loss + self.tau * F.cross_entropy(self.model(X_mem), y_mem)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MLP_GRAD_CLIP)
                self.optimizer.step()
                batch_loss += loss.item()
            scheduler.step()
            epoch_losses.append(batch_loss / max(1, len(loader)))
            if has_converged(epoch_losses):
                break

    def predict(self, X):
        return predict(self.model, X, self.device)

