"""
fsw.py — FSW: Fair Class-Incremental Learning using Sample Weighting
         (Park, Kim & Whang, KDD 2026; reference code github.com/jyparkkr/FSW).

Faithful reimplementation of the reference EO variant (binary sensitive
attribute) on this repo's training conventions. Once per epoch, before
training, an LP assigns weights w in [0,1] to every current-task sample so
that the first-order (last-layer) approximation of each (class, group) loss
after the epoch's update

    l~(G) = l(G) - alpha * < g_G_hat, sum_i w_i * g_i_hat >

minimizes the within-class group disparity sum_y |l~(G_{y,0}) - l~(G_{y,1})|/2
plus lambda * the mean approximated current-class loss (the accuracy
regularizer; lambda_old covers past classes and is 0 in the reference Bios
scripts). Both group and per-sample gradients are L2-normalized and the
learning rate is absorbed into the tuned scale alpha, as in the reference.
Samples with weight < 0.05 are dropped for the epoch. Replay follows the
paper's update g <- g_curr + tau * g_mem, implemented as an added tau-weighted
CE term on a memory batch (identical gradient), drawing AGEM_MEM_BATCH_SIZE
samples per step from the same EpisodicMemory budget as every other method.

Last-layer gradients are computed in closed form — for softmax CE the final
Linear's gradient for sample i is [b_i, b_i h_i^T] with b_i = p_i - onehot(y_i)
and h_i the penultimate activation — so the LP inputs need no autograd and the
(n_samples x grad_dim) matrix is never materialized (factored inner products).
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog
from torch.utils.data import DataLoader, TensorDataset

from model import (MLP, BertMLP, make_model, make_optimizer, make_scheduler,
                   has_converged, predict, mask_logits, _x_tensor,
                   _train_batch_size, batchnorm_eval)
from memory import EpisodicMemory
from config import (MLP_EPOCHS, MLP_GRAD_CLIP, DEVICE, AGEM_MEM_BATCH_SIZE, SEED,
                    FSW_ALPHA, FSW_LAMBDA, FSW_LAMBDA_OLD, FSW_TAU,
                    FSW_DROP_THRESHOLD)


class FSW:
    """Fairness-aware Sample Weighting (EO objective, |Z| = 2)."""

    def __init__(self, input_dim: int, device: str = DEVICE, bert_mode: bool = False,
                 bert_lr: float = None, alpha: float = FSW_ALPHA,
                 lmbd: float = FSW_LAMBDA, lmbd_old: float = FSW_LAMBDA_OLD,
                 tau: float = FSW_TAU):
        self.input_dim = input_dim
        self.device    = device
        self.bert_mode = bert_mode
        self.bert_lr   = bert_lr
        self.alpha     = alpha
        self.lmbd      = lmbd
        self.lmbd_old  = lmbd_old
        self.tau       = tau
        self.model     = make_model(input_dim, bert_mode).to(device)
        self.optimizer = make_optimizer(self.model, bert_lr=bert_lr)
        self.memory    = EpisodicMemory(device=device)
        self._pad_rng  = np.random.default_rng(SEED)
        self.weight_stats = {}   # task_idx -> per-epoch weight diagnostics

    def remember(self, X: np.ndarray, y: np.ndarray, s: np.ndarray):
        self.memory.add_task(X, y, s)

    # -- last-layer closed-form stats -------------------------------------

    def _final_linear(self) -> nn.Linear:
        return (self.model.head if isinstance(self.model, BertMLP) else self.model.net)[-1]

    def _penultimate(self, xb: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, BertMLP):
            return self.model.head[:-1](self.model.encode(xb))
        return self.model.net[:-1](xb)

    @torch.no_grad()
    def _stats(self, X_t: torch.Tensor, y_t: torch.Tensor,
               seen_classes: list[int] | None, batch_size: int = 1024):
        """Per-sample loss, b_i = p_i - onehot(y_i), and penultimate h_i.

        Eval mode (deterministic measurement pass, as in the reference, which
        runs its gradient pass under backbone.eval()).
        """
        was_training = self.model.training
        self.model.eval()
        losses, Bs, Hs = [], [], []
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(self.device)
            yb = y_t[i:i + batch_size].to(self.device)
            h  = self._penultimate(xb)
            logits = mask_logits(self._final_linear()(h), seen_classes)
            losses.append(F.cross_entropy(logits, yb, reduction="none").cpu())
            p = F.softmax(logits, dim=1)
            b = p - F.one_hot(yb, num_classes=logits.shape[1]).to(p.dtype)
            Bs.append(b.cpu())
            Hs.append(h.cpu())
        self.model.train(was_training)
        return torch.cat(losses), torch.cat(Bs), torch.cat(Hs)

    # -- LP construction ----------------------------------------------------

    @staticmethod
    def _group_vectors(loss, B, H, y, s, classes):
        """Mean loss and factored mean gradient (beta [C], W [CxE]) per (class, z).

        Group gradient = mean_i [b_i, b_i h_i^T] = [mean b_i, (B^T H)/n].
        Returns dict {(c, z): (mean_loss, beta, W)}; empty cells are absent.
        """
        out = {}
        for c in classes:
            for z in (0, 1):
                m = (y == c) & (s == z)
                n = int(m.sum())
                if n == 0:
                    continue
                Bg, Hg = B[m], H[m]
                out[(c, z)] = (loss[m].mean().item(), Bg.mean(0), Bg.T @ Hg / n)
        return out

    def _compute_weights(self, X_t, y_t, s_t, seen_classes, current_classes):
        """Solve the reference EO LP; returns per-sample weights (numpy, len(X_t))."""
        if self.alpha == 0:   # reference early-exit: no gradient term, plain training
            return np.ones(len(X_t), dtype=np.float32)
        mem = self.memory.get_all()
        X_m, y_m, s_m = mem
        loss_m, B_m, H_m = self._stats(X_m.cpu(), y_m.cpu(), seen_classes)
        loss_c, B_c, H_c = self._stats(X_t, y_t, seen_classes)

        # Group stats over buffer ∪ current. In class-IL the class sets are
        # disjoint, so this equals the reference (old classes from memory,
        # current classes from task data); in domain-IL (spurious_shift) both
        # sources share the label space and the group loss covers history too.
        y_all = torch.cat([y_m.cpu(), y_t])
        s_all = torch.cat([s_m.cpu(), s_t])
        loss_all = torch.cat([loss_m, loss_c])
        B_all, H_all = torch.cat([B_m, B_c]), torch.cat([H_m, H_c])
        old_classes = sorted(set(torch.unique(y_all).tolist()) - set(current_classes))
        groups = self._group_vectors(loss_all, B_all, H_all, y_all, s_all,
                                     old_classes + list(current_classes))

        # Keep classes with both (z=0, z=1) cells present; reference errors on
        # missing cells, we drop the class row and warn instead.
        ordered = [c for c in old_classes + list(current_classes)
                   if (c, 0) in groups and (c, 1) in groups]
        dropped = [c for c in old_classes + list(current_classes) if c not in ordered]
        if dropped:
            print(f"    [fsw] classes without both gender cells, excluded from LP: {dropped}")
        n = len(ordered)
        n_cur = sum(1 for c in ordered if c in set(current_classes))
        n_old = n - n_cur
        if n == 0 or n_cur == 0:
            return np.ones(len(X_t), dtype=np.float32)

        # Normalized per-sample gradient inner products, factored:
        # <[beta, W], g_i> = beta . b_i + b_i^T (W h_i);  ||g_i|| = ||b_i|| sqrt(1+||h_i||^2)
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
        C = torch.stack(C_rows).double().numpy()
        d = np.asarray(d_vec, dtype=np.float64)
        # lambda scaling; old classes get lambda_old (reference: /(n - n_cur), /n_cur)
        C, d = self.lmbd * C, self.lmbd * d
        if n_old > 0:
            C[:n_old] *= self.lmbd_old / n_old
            d[:n_old] *= self.lmbd_old / n_old
        C[n_old:] /= n_cur
        d[n_old:] /= n_cur

        return self._solve_lp(A, b, C, d)

    @staticmethod
    def _solve_lp(A, b, C, d):
        """min_x sum|Ax - b| + sum(d - Cx)  s.t.  0<=x<=1, Cx <= d.

        Reference formulation (cplex_solver.absolute_and_nonabsolute_minsum_
        LP_solver_v1): slack u_j = |A_j x - b_j| via two inequalities, and the
        non-absolute slack yy = d - Cx >= 0 makes Cx <= d a hard constraint.
        """
        K, m = A.shape
        # variables: [x (m), u (K)]
        cost = np.concatenate([-C.sum(0), np.ones(K)])
        A_ub = sp.vstack([
            sp.hstack([sp.csr_matrix(A),  -sp.eye(K)]),   # Ax - u <= b
            sp.hstack([sp.csr_matrix(-A), -sp.eye(K)]),   # -Ax - u <= -b
            sp.hstack([sp.csr_matrix(C),  sp.csr_matrix((K, K))]),  # Cx <= d
        ], format="csr")
        b_ub = np.concatenate([b, -b, d])
        res = linprog(cost, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(0.0, 1.0)] * m + [(0.0, None)] * K,
                      method="highs")
        if not res.success:
            print(f"    [fsw] LP failed ({res.message}); using uniform weights")
            return np.ones(m, dtype=np.float32)
        return res.x[:m].astype(np.float32)

    # -- training -------------------------------------------------------------

    def train_task(self, X_train: np.ndarray, y_train: np.ndarray,
                   s_train: np.ndarray = None, epochs: int = MLP_EPOCHS,
                   seen_classes: list[int] | None = None):
        if s_train is None:
            s_train = np.zeros(len(y_train), dtype=np.int64)
        X_t = _x_tensor(X_train)
        y_t = torch.tensor(y_train, dtype=torch.long)
        s_t = torch.tensor(s_train, dtype=torch.long)
        task_idx = self.memory.num_tasks()
        current_classes = sorted(np.unique(y_train).tolist())
        batch_size = _train_batch_size(self.bert_mode)
        n_batches_full = int(np.ceil(len(X_t) / batch_size))

        scheduler = make_scheduler(self.optimizer, epochs, steps_per_epoch=n_batches_full)
        gen = torch.Generator()
        # KNOWN DEVIATION: seeded from the fixed config SEED, whereas every other
        # method seeds this generator from runctx.effective_seed() so --seed
        # varies batch order. FSW's batch order is therefore identical across
        # seeds, and its cross-seed spread omits that one source of variation
        # (model init, the validation split and memory sampling still vary,
        # via the global RNG runctx.apply_seed sets). The reported FSW results
        # were produced with this seeding.
        gen.manual_seed(SEED)
        criterion = nn.CrossEntropyLoss(reduction="none")
        self.weight_stats[task_idx] = []

        epoch_losses = []
        for epoch in range(epochs):
            # Task 0 trains plainly (no memory, no weighting), as in the reference.
            drop_last = False
            if task_idx > 0:
                w = self._compute_weights(X_t, y_t, s_t, seen_classes, current_classes)
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
                pad = (-len(idx)) % batch_size
                if pad:
                    pool = np.where(~keep)[0]
                    take = min(pad, len(pool))
                    if take:
                        idx = np.concatenate([idx, self._pad_rng.choice(pool, take, replace=False)])
                drop_last = len(idx) % batch_size == 1 and len(idx) > batch_size
                idx_t = torch.from_numpy(idx)
                w_t = torch.tensor(w, dtype=torch.float32)
                ds = TensorDataset(X_t[idx_t], y_t[idx_t], w_t[idx_t])
            else:
                ds = TensorDataset(X_t, y_t, torch.ones(len(X_t)))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen,
                                drop_last=drop_last)

            self.model.train()
            batch_loss = 0.0
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(self.device), yb.to(self.device), wb.to(self.device)
                self.optimizer.zero_grad()
                loss = (criterion(mask_logits(self.model(xb), seen_classes), yb) * wb).mean()
                if task_idx > 0 and self.tau > 0:
                    mem = self.memory.get_random_batch(AGEM_MEM_BATCH_SIZE)
                    if mem is not None:
                        X_mem, y_mem, _ = mem
                        with batchnorm_eval(self.model):
                            mem_loss = F.cross_entropy(
                                mask_logits(self.model(X_mem), seen_classes), y_mem)
                        loss = loss + self.tau * mem_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MLP_GRAD_CLIP)
                self.optimizer.step()
                scheduler.step()
                batch_loss += loss.item()
            epoch_losses.append(batch_loss / max(1, len(loader)))
            if has_converged(epoch_losses):
                break

    def predict(self, X: np.ndarray, seen_classes: list[int] | None = None) -> np.ndarray:
        return predict(self.model, X, self.device, seen_classes=seen_classes)

