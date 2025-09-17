# supersmooth2_with_soft_superpixels_and_demo_profiled_fast_rag.py
# Final version: decoupled SLIC vs SOFT compactness + color_weight in distances
# (Fixed: blend in linear RGB to avoid Lab brightening; α used; soft-compactness in w; robust softmax w/o torch.nan*)

import os
import math
import time
import shutil
from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm, binomtest
from statsmodels.stats.proportion import proportion_confint
from skimage.segmentation import felzenszwalb, quickshift, mark_boundaries
from skimage import color  # kept for fallback/compat
import matplotlib.pyplot as plt
from datasets import get_dataset
from architectures import get_architecture

# Try AVX2-optimized fast_slic; fall back to standard fast_slic
try:
    from fast_slic.avx2 import SlicAvx2 as _SlicImpl
except Exception:
    from fast_slic import Slic as _SlicImpl

_DEMO_SOFT_COUNTER = 0  # incremented each time _sample_noise is entered



class SuperSmoothSoft(object):
    """A smoothed classifier g (with optional soft superpixels, profiling, and fast RAG)."""

    ABSTAIN = -1

    def __init__(self,
            base_classifier: torch.nn.Module,
            num_classes: int,
            sigma: float,
            k: int,
            # --- decoupled compactness knobs ---
            slic_compactness: float = 10.0,   # used ONLY for SLIC label generation
            soft_compactness: float = 5.0,   # used ONLY inside soft distance: w=(c/S)^2
            c: float = None,
            ns: int = 2000,
            alg: str = 'soft',
            channel: int = 3,
            tau: float = 4.0,                 # softmax temperature for 'soft'
            device: str = 'cuda:0',
            noise_factor: float = 1.0,
            topk: int = 4,
            save_demos: bool = False,
            demo_every: int = 5,
            soft_max_neighbors: int = 6,
            color_weight: float = 0.7,       # α in D = α*dc2 + β*w*ds2
        ):
        
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device
        self.kernel = construct_kernel(k).to(self.device)
        if channel == 1:
            self.kernel = construct_kernel_mnist(k).to(self.device)

        self.slic_compactness = slic_compactness
        self.soft_compactness = soft_compactness
        self.num_superpixels = ns
        self.alg = alg
        self.channel = channel
        self.noise_factor = noise_factor

        # soft-superpixel knobs
        self.tau = tau
        self.topk = topk
        self.soft_max_neighbors = int(soft_max_neighbors)
        self.color_weight = float(color_weight)

        # demos
        self.save_demos = bool(save_demos)
        self.saved = False
        self.demo_every = int(demo_every) if int(demo_every) > 0 else 5

        self._ensure_demo_dir()
        torch.backends.cudnn.benchmark = True

    # ---------------------------
    # Public API
    # ---------------------------
    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, true_label) -> (int, float):
        self.base_classifier.eval()
        if self.save_demos:
            self.saved=False
            
        
        counts_selection = self._sample_noise(x, n0, batch_size)
        cAHat = counts_selection.argmax().item()
        if cAHat != true_label:
            return cAHat, 0.0
        counts_estimation = self._sample_noise(x, n, batch_size, debug=False)
        counts_estimation = counts_estimation + counts_selection
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n + n0, alpha)
        if pABar < 0.5:
            return self.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar) / self.noise_factor
            return cAHat, float(radius)

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binomtest(count1, count1 + count2, p=0.5).pvalue > alpha:
            return self.ABSTAIN
        else:
            return int(top2[0])

    # ---------------------------
    # Sampling
    # ---------------------------
    def _sample_noise(self, x: torch.tensor, num: int, batch_size, debug=False) -> np.ndarray:
        save_this_call = False
        save_index = None
        if self.save_demos and self.alg in ['soft', 'static']:
            global _DEMO_SOFT_COUNTER
            _DEMO_SOFT_COUNTER += 1
            if _DEMO_SOFT_COUNTER % self.demo_every == 0:
                save_this_call = True
                save_index = _DEMO_SOFT_COUNTER

        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                batch = x.repeat((this_batch_size, 1, 1, 1))

                z1 = torch.randn_like(batch, device=self.device)
                z2 = torch.randn_like(batch, device=self.device)

                rho = 1 / self.noise_factor
                sqrt_term = math.sqrt(max(0.0, 1.0 - rho * rho))
                noise = self.sigma * z1
                noise_tau = self.sigma * ((rho / self.noise_factor) * z1 + (sqrt_term / self.noise_factor) * z2)

                # control (denoised) for SLIC labels
                conv_images = convolve_batch(batch + noise_tau, self.kernel)

                if self.alg == 'soft':
                    superpixel_images, labels = self.batch_superpixel_soft(
                        conv_images, batch + noise,
                        num_superpixels=self.num_superpixels,
                        tau=self.tau,
                        topk=self.topk,
                    )

                    if self.save_demos and save_this_call:
                        try:
                            self._save_soft_visuals_periodic(
                                noisy_image_t=(batch + noise)[0],
                                c_image_t=conv_images[0],
                                labels_t=labels[0],
                                soft_image_t=superpixel_images[0],
                                outfile_base=os.path.join("demo", "soft_superpixel"),
                                save_index=save_index
                            )
                        except Exception:
                            pass

                else:
                    superpixel_images, labels = self.batch_superpixel_average(
                        conv_images, batch + noise,
                        num_superpixels=self.num_superpixels,
                        alg=self.alg
                    )

                    if self.save_demos and save_this_call and self.alg == 'static':
                        try:
                            self._save_static_visuals_periodic(
                                noisy_image_t=(batch + noise)[0],
                                c_image_t=conv_images[0],
                                labels_t=labels[0],
                                static_image_t=superpixel_images[0],
                                outfile_base=os.path.join("demo2", "static_superpixel"),
                                save_index=save_index
                            )
                        except Exception:
                            pass

                predictions = self.base_classifier(superpixel_images).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    @staticmethod
    def _count_arr(arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    @staticmethod
    def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
        return float(proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0])

    # ---------------------------
    # Superpixels (fast_slic / fz / qs)
    # ---------------------------
    def superpixel(self, image, num_superpixels, compactness=10.0, **_):
        img = image.detach().to('cpu')
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img_u8 = np.ascontiguousarray(
            (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        )
        slic_obj = _SlicImpl(num_components=num_superpixels, compactness=compactness)
        labels = slic_obj.iterate(img_u8)
        return labels

    def superpixel_fz(self, image, num_superpixels, compactness=10.0, max_iterations=10, sigma=0.7):
        img = image.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        labels = felzenszwalb(img, scale=5, sigma=0.2, min_size=20)
        return labels

    def superpixel_qs(self, image, num_superpixels, compactness=10.0, max_iterations=10, sigma=0.7):
        img = image.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        labels = quickshift(img, kernel_size=3, max_dist=4, ratio=0.5)
        return labels

    def superpixel_static(self, image, num_superpixels=None, *, cell_h=4, cell_w=4, **_):
        """
        Static rectangular grid with fixed patch size (cell_h x cell_w).
        - Exact tiling with full patches only (no stretching at image boundaries).
        - Any leftover border pixels are ignored and labeled -1.
        - 'num_superpixels' is unused (kept for API compatibility).
        """
        import numpy as np

        _, H, W = image.shape

        # How many full patches fit without crossing the boundary
        best_rows = H // cell_h
        best_cols = W // cell_w

        labels = -np.ones((H, W), dtype=np.int32)  # ignored pixels stay -1

        label_id = 0
        for row in range(best_rows):
            for col in range(best_cols):
                y_start = row * cell_h
                y_end   = y_start + cell_h   # do NOT extend to H
                x_start = col * cell_w
                x_end   = x_start + cell_w   # do NOT extend to W

                labels[y_start:y_end, x_start:x_end] = label_id
                label_id += 1

        return labels

    # ---------------------------
    # Optimized average per segment (GPU) — HARD path
    # ---------------------------
    def average_each_segment_torch(self, image: torch.Tensor, labels_np: np.ndarray,
                                   _profile_sink: dict = None, _use_sync: bool = False) -> torch.Tensor:
        device = image.device
        C, H, W = image.shape
        labels = torch.from_numpy(labels_np.astype(np.int64, copy=False)).to(device)
        labels_flat = labels.view(-1)           # [HW]
        pixels = image.view(C, -1).t()          # [HW, C]
        L = int(labels_flat.max().item()) + 1

        sums = torch.zeros((L, C), device=device, dtype=image.dtype)
        sums.index_add_(0, labels_flat, pixels)         # sum per label
        counts = torch.bincount(labels_flat, minlength=L).clamp_min(1).unsqueeze(1).to(image.dtype)
        means = sums / counts
        
        out = means[labels_flat]                        # [HW, C]
        out = out.t().contiguous().view(C, H, W)
        return out

    def batch_superpixel_average(self, c_images, images, num_superpixels, alg='slic'):
        batch_size = images.shape[0]
        averaged_images = torch.zeros_like(images)
        labels_tensor = torch.empty((images.shape[0], images.shape[2], images.shape[3]),
                                    device=images.device, dtype=torch.long)

        for i in range(batch_size):
            if alg == 'slic':
                label = self.superpixel(c_images[i], num_superpixels, compactness=self.slic_compactness)
            elif alg == 'fz':
                label = self.superpixel_fz(c_images[i], num_superpixels)
            elif alg == 'qs':
                label = self.superpixel_qs(c_images[i], num_superpixels)
            elif alg == 'static':
                label = self.superpixel_static(c_images[i], num_superpixels)
            else:
                label = self.superpixel(c_images[i], num_superpixels, compactness=self.slic_compactness)

            out = self.average_each_segment_torch(images[i], label, None, False)

            averaged_images[i] = out
            labels_tensor[i] = torch.from_numpy(label.astype(np.int64, copy=False)).to(images.device)

        return averaged_images, labels_tensor

    
    def _build_cand_mat_torch(self, labels_t: torch.Tensor, max_neighbors: int = 12,
                          include_self: bool = True) -> torch.Tensor:
        device = labels_t.device
        H, W = labels_t.shape
        K = int(labels_t.max().item()) + 1
        M = (1 if include_self else 0) + max_neighbors
        adj = torch.zeros((K, K), dtype=torch.bool, device=device)
        # Horizontal edges
        a = labels_t[:, :-1]
        b = labels_t[:, 1:]
        mask = a != b
        ar = a[mask]
        br = b[mask]
        adj[ar, br] = True
        adj[br, ar] = True
        # Vertical edges
        a = labels_t[:-1, :]
        b = labels_t[1:, :]
        mask = a != b
        ad = a[mask]
        bd = b[mask]
        adj[ad, bd] = True
        adj[bd, ad] = True
        # Nonzero indices
        nz = torch.nonzero(adj, as_tuple=False)  # [num_nz, 2]
        # Sort by row then col
        sort_keys = nz[:, 0] * K + nz[:, 1]
        sort_idx = torch.argsort(sort_keys)
        nz_sorted = nz[sort_idx]
        rows = nz_sorted[:, 0]
        cols = nz_sorted[:, 1]
        # Degrees (neighbors per row)
        degrees = torch.bincount(rows, minlength=K)
        # Offsets (prefix sums)
        offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(degrees, dim=0)])
        # Candidate matrix
        cand = torch.full((K, M), -1, dtype=torch.long, device=device)
        start_col = 0
        if include_self:
            cand[:, 0] = torch.arange(K, device=device)
            start_col = 1
        # Vectorized fill of neighbors
        for i in range(max_neighbors):
            mask = degrees > i
            if not mask.any():
                break
            sel_rows = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            offs = offsets[:-1][sel_rows] + i
            cand[sel_rows, start_col + i] = cols[offs]
        return cand


    # ---- GPU Lab/XYZ helpers ----
    def _rgb_to_lab_torch(self, rgb_hw3: torch.Tensor) -> torch.Tensor:
        srgb = rgb_hw3.clamp(0, 1)
        thr = 0.04045
        lin = torch.where(srgb <= thr, srgb / 12.92, ((srgb + 0.055) / 1.055).pow(2.4))
        M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]], device=lin.device, dtype=lin.dtype)
        XYZ = lin @ M.T
        Xn = torch.tensor([0.95047, 1.00000, 1.08883], device=lin.device, dtype=lin.dtype)
        xr = XYZ / Xn
        eps = 216/24389
        k = 24389/27
        def f(t):
            t_cbrt = torch.pow(t.clamp(min=0), 1/3)
            return torch.where(t > eps, t_cbrt, (k * t + 16.) / 116.)
        fx, fy, fz = f(xr[..., 0]), f(xr[..., 1]), f(xr[..., 2])
        L = 116*fy - 16
        a = 500*(fx - fy)
        b = 200*(fy - fz)
        return torch.stack([L, a, b], dim=-1)


    def _srgb_to_linear(self, srgb_hw3: torch.Tensor) -> torch.Tensor:
        thr = 0.04045
        return torch.where(srgb_hw3 <= thr, srgb_hw3 / 12.92,
                           ((srgb_hw3 + 0.055) / 1.055).pow(2.4))

    def _linear_to_srgb(self, lin_hw3: torch.Tensor) -> torch.Tensor:
        thr = 0.0031308
        srgb = torch.where(lin_hw3 <= thr, 12.92 * lin_hw3,
                           1.055 * lin_hw3.clamp(min=0).pow(1/2.4) - 0.055)
        return srgb.clamp(0, 1)

    def _get_yx_grid(self, H: int, W: int, device, dtype=torch.float32):
        key = (H, W, str(device))
        cache = getattr(self, "_yx_cache", None)
        if cache is None:
            self._yx_cache = {}
            cache = self._yx_cache
        if key not in cache:
            ys = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
            xs = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
            cache[key] = torch.stack([ys, xs], dim=-1).reshape(-1, 2).contiguous()
        return cache[key]


    def batch_superpixel_soft(self,
                            c_images: torch.Tensor,
                            images: torch.Tensor,
                            num_superpixels: int,
                            compactness: float = 20.0,
                            tau: float = 0.5,
                            topk: int = 5):
        batch_size, C, H, W = images.shape
        blended = torch.zeros_like(images)
        labels_tensor = torch.empty((batch_size, H, W), device=images.device, dtype=torch.long)

        for i in range(batch_size):
            timings = {}

            # 1) SLIC on the control image
            labels_np = self.superpixel(c_images[i], num_superpixels, compactness=self.slic_compactness)
            labels_tensor[i] = torch.from_numpy(labels_np.astype(np.int64, copy=False)).to(images.device)

            # ---- A) Distances from image (stable features) ----
            img_for_dist = c_images[i]
            if img_for_dist.shape[0] == 1:
                img_for_dist = img_for_dist.repeat(3, 1, 1)
            rgb_hw3_dist = img_for_dist.permute(1, 2, 0).contiguous()
            lab_hw3_dist = self._rgb_to_lab_torch(rgb_hw3_dist)
            lab_flat_dist_t = lab_hw3_dist.reshape(-1, 3)
            # ---- B) Centers from NOISY image (segment-mean only) ----
            img_for_centers = images[i]
            if img_for_centers.shape[0] == 1:
                img_for_centers = img_for_centers.repeat(3, 1, 1)
            rgb_hw3_cent = img_for_centers.permute(1, 2, 0).contiguous()
            lab_hw3_cent = self._rgb_to_lab_torch(rgb_hw3_cent)
            lab_flat_cent_t = lab_hw3_cent.reshape(-1, 3)
            # 3) centers (Lab & XY) computed from the NOISY Lab + robust sRGB centers for blending 
            yx_t = self._get_yx_grid(H, W, img_for_centers.device)
            labels_flat = labels_tensor[i].view(-1)
            K = int(labels_flat.max().item()) + 1
            counts = torch.bincount(labels_flat, minlength=K).clamp_min(1).to(lab_flat_cent_t.dtype).unsqueeze(1)
            #centers from NOISY stats (Lab for distance)
            sums_lab = torch.zeros((K, 3), device=img_for_centers.device, dtype=lab_flat_cent_t.dtype)
            sums_lab.index_add_(0, labels_flat, lab_flat_cent_t)
            centers_lab_t = sums_lab / counts
            sums_xy = torch.zeros((K, 2), device=img_for_centers.device, dtype=lab_flat_cent_t.dtype)
            sums_xy.index_add_(0, labels_flat, yx_t)
            centers_xy_t = sums_xy / counts
            #robust centers in sRGB from NOISY image (winsorized mean) ----
            srgb_hw3_noisy = img_for_centers.permute(1, 2, 0).clamp(0, 1).contiguous()
            srgb_flat = srgb_hw3_noisy.view(-1, 3)
            sums_srgb = torch.zeros((K, 3), device=img_for_centers.device, dtype=srgb_flat.dtype)
            sums_srgb.index_add_(0, labels_flat, srgb_flat)
            means_srgb = sums_srgb / counts  # [K,3]
            sumsq_srgb = torch.zeros((K, 3), device=img_for_centers.device, dtype=srgb_flat.dtype)
            sumsq_srgb.index_add_(0, labels_flat, srgb_flat * srgb_flat)
            var_srgb = (sumsq_srgb / counts - means_srgb * means_srgb).clamp_min(0.0)
            std_srgb = var_srgb.sqrt() + 1e-6
            mu = means_srgb[labels_flat]         # [HW,3]
            sg = std_srgb[labels_flat]           # [HW,3]
            t_clip = 1.5
            clipped = mu + (srgb_flat - mu).clamp(-t_clip * sg, t_clip * sg)
            cent_sums = torch.zeros((K, 3), device=img_for_centers.device, dtype=srgb_flat.dtype)
            cent_sums.index_add_(0, labels_flat, clipped)
            centers_srgb_t = cent_sums / counts  # [K,3] (colors depend only on noisy image)
            cand_mat_t = self._build_cand_mat_torch(
                labels_tensor[i], max_neighbors=self.soft_max_neighbors, include_self=True
            )

            # 5) Candidate gather
            cand_idx = cand_mat_t[labels_flat]
            valid = cand_idx >= 0
            safe_idx = torch.where(valid, cand_idx, torch.zeros_like(cand_idx))
            C_lab = centers_lab_t[safe_idx]
            C_xy  = centers_xy_t[safe_idx]
            lab_e = lab_flat_dist_t.unsqueeze(1)     
            yx_e  = yx_t.unsqueeze(1)
            # 6+7) Distances + per-pixel normalized softmax (no torch.nan*)
            # ----- grid scale from number of labels -----
            S = math.sqrt((H * W) / max(K, 1))
            # use SOFT compactness here (decoupled from SLIC)
            w = (float(self.soft_compactness) / max(S, 1e-12))**2
            alpha = float(self.color_weight)          # global color weight
            tau_val = max(float(tau), 1e-12)
            # raw squared distances
            dc2 = torch.sum((lab_e - C_lab)**2, dim=2)  # [HW, M]
            ds2 = torch.sum((yx_e  - C_xy )**2, dim=2)  # [HW, M]
            # ---- robust scalar balance between color & spatial over VALID entries only (no NaNs) ----
            dc2_flat  = dc2.masked_select(valid)
            ds2_flat  = ds2.masked_select(valid)
            if dc2_flat.numel() == 0:
                m_dc = torch.tensor(1.0, device=dc2.device, dtype=dc2.dtype)
                m_ds = torch.tensor(1.0, device=dc2.device, dtype=dc2.dtype)
            else:
                m_dc = dc2_flat.median()
                m_ds = (w * ds2_flat).median()
            beta = (m_dc / m_ds).clamp(0.3, 3.0).detach()
            # final distance (mask invalid to +inf)
            D = alpha * dc2 + beta * w * ds2
            # print(f'alpha * dc2: {alpha*dc2}, beta * w * ds2: {beta*w*ds2}')
            # print(f'beta: {beta}, w: {w}, alpha:{w}')
            inf = torch.full_like(D, float('inf'))
            D_valid = torch.where(valid, D, inf)        # [HW, M] finite on valid, +inf on invalid
            # ---- per-pixel normalization to avoid one-hot P ----
            # shift: best candidate has 0
            Dmin = torch.min(D_valid, dim=1, keepdim=True).values
            D0 = D_valid - Dmin                          # >=0 on valid, inf on invalid
            # robust per-pixel scale: median of positive gaps only
            pos_mask = valid & (D0 > 0)
            D0_pos = torch.where(pos_mask, D0, inf)
            n_pos = pos_mask.sum(dim=1)                  # [HW]
            has_pos = n_pos > 0
            D0_sorted, _ = torch.sort(D0_pos, dim=1)     # [HW, M], infs go to the end
            k_med = torch.clamp((n_pos - 1) // 2, min=0)  # [HW]
            scale_med = D0_sorted.gather(1, k_med.view(-1,1))  # [HW,1]
            one = torch.ones_like(scale_med)
            scale = torch.where(has_pos.view(-1,1), scale_med, one).clamp_min(1e-6)
            # softmax over normalized distances (invalid -> -inf)
            neg = -(D0 / (tau_val * scale))
            neg = torch.where(valid, neg, torch.full_like(neg, float('-inf')))
            mmax = torch.max(neg, dim=1, keepdim=True).values
            exps = torch.exp(neg - mmax)                 # exp(-inf)=0 for invalid
            Z = exps.sum(dim=1, keepdim=True) + 1e-12
            P = exps / Z
            # Top-K probabilities & indices among candidates
            k_sel = min(int(topk), P.shape[1])
            Pk, part = torch.topk(P, k_sel, dim=1)
            Pk = Pk / torch.sum(Pk, dim=1, keepdim=True)
            Ck = torch.gather(cand_idx, 1, part)
            # 8) Blend in sRGB using noisy-only robust centers (kept timer name)
            C_srgb_sel = centers_srgb_t[Ck]                          # [HW,k,3] in sRGB [0,1]
            C_lin_sel  = self._srgb_to_linear(C_srgb_sel)            # --> linear light
            out_lin    = torch.sum(Pk.unsqueeze(-1) * C_lin_sel, dim=1)  # [HW,3] linear avg
            out_srgb   = self._linear_to_srgb(out_lin)               # back to sRGB
            out_srgb_hw3 = out_srgb.reshape(H, W, 3).contiguous()
            out_i = out_srgb_hw3.permute(2, 0, 1).contiguous()

            blended[i] = out_i if C == 3 else out_i.mean(0, keepdim=True)


        return blended.clamp(0, 1), labels_tensor


    # -----------------------------------------------------------------------------
    # Demo-saving helpers
    # -----------------------------------------------------------------------------
    def _ensure_demo_dir(self, reset: bool = True):
        demo_dir = os.path.abspath("demo2")
        try:
            if reset and os.path.isdir(demo_dir):
                if demo_dir in ("/", os.path.expanduser("~")):
                    raise RuntimeError(f"Refusing to reset dangerous path: {demo_dir}")
                shutil.rmtree(demo_dir, ignore_errors=True)
            os.makedirs(demo_dir, exist_ok=True)
        except Exception:
            pass

    def _save_soft_visuals_periodic(self,
                                    noisy_image_t: torch.Tensor,
                                    c_image_t: torch.Tensor,
                                    labels_t: torch.Tensor,
                                    soft_image_t: torch.Tensor,
                                    outfile_base: str,
                                    save_index: int = None):
        if not self.save_demos or self.saved:
            return
        try:
            noisy_np = noisy_image_t.detach().to('cpu').permute(1, 2, 0).clamp(0, 1).numpy()
            labels_np = labels_t.detach().to('cpu').numpy()
            soft_np   = soft_image_t.detach().to('cpu').permute(1, 2, 0).clamp(0, 1).numpy()
            with_bounds = mark_boundaries(noisy_np, labels_np, mode='thick')

            # hard averaged using same labels (cheap)
            hard_avg_t = self.average_each_segment_torch(noisy_image_t, labels_np)
            hard_avg_np = hard_avg_t.detach().to('cpu').permute(1, 2, 0).clamp(0, 1).numpy()

            suffix = f"_{save_index}" if save_index is not None else ""

            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(noisy_np);      axs[0].set_title('Noisy'); axs[0].axis('off')
            axs[1].imshow(with_bounds);   axs[1].set_title('Hard SLIC (boundaries)'); axs[1].axis('off')
            axs[2].imshow(hard_avg_np);   axs[2].set_title('Hard superpixel (averaged)'); axs[2].axis('off')
            axs[3].imshow(soft_np);       axs[3].set_title('Soft superpixel (smoothed)'); axs[3].axis('off')
            plt.tight_layout()
            plt.savefig(f"{outfile_base}_demo{suffix}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(noisy_np);    axs[0].set_title('Noisy'); axs[0].axis('off')
            axs[1].imshow(with_bounds); axs[1].set_title('Hard SLIC (boundaries)'); axs[1].axis('off')
            axs[2].imshow(soft_np);     axs[2].set_title('Soft superpixel (from method)'); axs[2].axis('off')
            plt.tight_layout()
            plt.savefig(f"{outfile_base}_from_method{suffix}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.saved = True
        except Exception as e:
            try:
                print(f"[soft visuals] save skipped: {e}")
            except:
                pass

    def _save_static_visuals_periodic(self,
                                     noisy_image_t: torch.Tensor,
                                     c_image_t: torch.Tensor,
                                     labels_t: torch.Tensor,
                                     static_image_t: torch.Tensor,
                                     outfile_base: str,
                                     save_index: int = None):
        if not self.save_demos or self.saved:
            return
        try:
            noisy_np = noisy_image_t.detach().to('cpu').permute(1, 2, 0).clamp(0, 1).numpy()
            labels_np = labels_t.detach().to('cpu').numpy()
            static_np = static_image_t.detach().to('cpu').permute(1, 2, 0).clamp(0, 1).numpy()
            with_bounds = mark_boundaries(noisy_np, labels_np, mode='thick')

            suffix = f"_{save_index}" if save_index is not None else ""

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(noisy_np);      axs[0].set_title('Noisy'); axs[0].axis('off')
            axs[1].imshow(with_bounds);   axs[1].set_title('Static Grid (boundaries)'); axs[1].axis('off')
            axs[2].imshow(static_np);     axs[2].set_title('Static superpixel (averaged)'); axs[2].axis('off')
            plt.tight_layout()
            plt.savefig(f"{outfile_base}_demo{suffix}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(noisy_np);    axs[0].set_title('Noisy'); axs[0].axis('off')
            axs[1].imshow(static_np);   axs[1].set_title('Static Grid (averaged)'); axs[1].axis('off')
            plt.tight_layout()
            plt.savefig(f"{outfile_base}_from_method{suffix}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.saved = True
        except Exception as e:
            try:
                print(f"[static visuals] save skipped: {e}")
            except:
                pass
    
    def sample_partioning(self, x, sigma):

        batch_size = 1
        noisy_batch = x.repeat((batch_size, 1, 1, 1))
        noise = torch.randn_like(noisy_batch) * sigma
        noisy_image = noisy_batch + noise
        c_image = convolve_batch(noisy_image, self.kernel)
        
        superpixel_images_soft, labels_soft = self.batch_superpixel_soft(
            c_image, noisy_image,
            num_superpixels=self.num_superpixels,
            tau=self.tau
        )
        superpixel_images_hard, labels_hard = self.batch_superpixel_average(
            c_image, noisy_image,
            num_superpixels=self.num_superpixels,
            alg='slic'
        )
        return superpixel_images_soft, labels_soft, superpixel_images_hard, labels_hard, noisy_image


def construct_kernel_mnist(k: int):
    kernel = torch.zeros((1, 1, k, k))
    for i in range(1):
        kernel[i, i, :, :] = torch.ones((k, k)) / (k * k)
    return kernel

def construct_kernel(k: int):
    kernel = torch.zeros((3, 3, k, k))
    for i in range(3):
        kernel[i, i, :, :] = torch.ones((k, k)) / (k * k)
    return kernel

def convolve_batch(imgs, kernel):
    return F.conv2d(imgs, kernel, padding='same')
