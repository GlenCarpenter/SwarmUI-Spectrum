# SwarmSpectrum.py
# Implements Spectrum: Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration
# Paper: https://arxiv.org/abs/2603.01623 (CVPR 2026, hanjq17 et al., Stanford / ByteDance)
# Supports: SDXL, Flux, Wan, SD3, HunyuanVideo, Chroma, and any model using ComfyUI's model patching API.

import math
import torch


class _ChebyshevForecaster:
    """Chebyshev polynomial + ridge regression feature forecaster.

    Maintains a sliding window of true denoiser output feature vectors and predicts
    future features using global spectral approximation blended with local Taylor
    extrapolation. Supports residual calibration for detail recovery.
    """

    def __init__(self, m: int = 4, lam: float = 0.1):
        self.M = m
        self.K = max(m + 2, 8)  # sliding window capacity
        self.lam = lam
        self.H_buf: list[torch.Tensor] = []  # feature history, flattened float32
        self.T_buf: list[float] = []          # normalized timestep history
        self.shape: torch.Size | None = None
        self.dtype: torch.dtype | None = None
        self.total_steps: int = 30
        # Calibrated-mode state
        self.residual: torch.Tensor | None = None
        self.last_raw_pred: torch.Tensor | None = None

    def _tau(self, step: int) -> float:
        """Map step index to Chebyshev coordinate in [-1, 1]."""
        total = max(self.total_steps, 1)
        return 2.0 * (step / total) - 1.0

    def _cheb_basis(self, taus: torch.Tensor) -> torch.Tensor:
        """Build Chebyshev polynomial design matrix of shape (N, M+1)."""
        taus = taus.reshape(-1, 1)
        cols = [torch.ones((taus.shape[0], 1), device=taus.device, dtype=torch.float32)]
        if self.M > 0:
            cols.append(taus)
            for _ in range(2, self.M + 1):
                cols.append(2.0 * taus * cols[-1] - cols[-2])
        return torch.cat(cols[:self.M + 1], dim=1)

    def update(self, step: int, h: torch.Tensor) -> None:
        """Record a true denoiser output at the given step index."""
        if self.shape is not None and h.shape != self.shape:
            self.reset()
        self.shape = h.shape
        self.dtype = h.dtype
        self.H_buf.append(h.detach().float().view(-1))
        self.T_buf.append(self._tau(step))
        if len(self.H_buf) > self.K:
            self.H_buf.pop(0)
            self.T_buf.pop(0)

    def predict(self, step: int, w: float, calibration_strength: float = 0.0) -> torch.Tensor:
        """Predict the denoiser output at the given step using Chebyshev + Taylor blend.

        Args:
            step: Current denoising step index.
            w: Blending weight: 0 = pure Taylor, 1 = pure Chebyshev.
            calibration_strength: If > 0, blend in the last residual correction.

        Returns:
            Predicted feature tensor with same shape/dtype as original denoiser output.
        """
        device = self.H_buf[-1].device
        H = torch.stack(self.H_buf, dim=0)  # (K, D)
        T = torch.tensor(self.T_buf, dtype=torch.float32, device=device)  # (K,)
        X = self._cheb_basis(T)  # (K, M+1)
        lam_I = self.lam * torch.eye(self.M + 1, device=device)
        XtX = X.T @ X + lam_I
        try:
            L = torch.linalg.cholesky(XtX)
        except RuntimeError:
            jitter = 1e-5 * XtX.diagonal().mean()
            L = torch.linalg.cholesky(XtX + jitter * torch.eye(self.M + 1, device=device))
        coef = torch.cholesky_solve(X.T @ H, L)  # (M+1, D)
        tau_star = torch.tensor([self._tau(step)], dtype=torch.float32, device=device)
        pred_cheb = (self._cheb_basis(tau_star) @ coef).squeeze(0)  # (D,)
        if len(self.H_buf) >= 2:
            h_taylor = self.H_buf[-1] + 0.5 * (self.H_buf[-1] - self.H_buf[-2])
        else:
            h_taylor = self.H_buf[-1]
        raw = (1.0 - w) * h_taylor + w * pred_cheb
        self.last_raw_pred = raw.detach().clone()
        if calibration_strength > 0.0 and self.residual is not None:
            raw = raw + self.residual.to(device=device, dtype=torch.float32) * calibration_strength
        return torch.clamp(raw, -10.0, 10.0).to(dtype=self.dtype).view(self.shape)

    def record_residual(self, true_output: torch.Tensor) -> None:
        """Store the difference between the true denoiser output and the last raw prediction."""
        if self.last_raw_pred is not None:
            self.residual = true_output.detach().float().view(-1) - self.last_raw_pred

    def reset(self) -> None:
        """Clear all buffered state."""
        self.H_buf.clear()
        self.T_buf.clear()
        self.shape = None
        self.dtype = None
        self.residual = None
        self.last_raw_pred = None


def _slice_batch(value, indices: list[int], batch_size: int):
    """Slice a tensor along dim 0 if it has a matching batch dimension, otherwise return as-is."""
    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size and len(indices) < batch_size:
        idx_tensor = torch.tensor(indices, device=value.device)
        return value[idx_tensor]
    return value


class SwarmSpectrum:
    """Applies Spectrum sampling acceleration to any ComfyUI diffusion model.

    Works by wrapping the model's denoiser function to skip selected timestep
    evaluations, replacing them with Chebyshev polynomial forecasts. Compatible
    with SDXL, Flux, Wan, SD3, HunyuanVideo, Chroma, and any model that uses
    ComfyUI's set_model_unet_function_wrapper API.

    Based on the CVPR 2026 paper:
      'Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration'
      Han et al., Stanford University & ByteDance
      https://arxiv.org/abs/2603.01623
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "w": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blending weight: 0 = pure local Taylor, 1 = pure global Chebyshev. 0.3-0.6 recommended.",
                }),
                "m": ("INT", {
                    "default": 3, "min": 1, "max": 8,
                    "tooltip": "Chebyshev polynomial basis count (forecast complexity). 3-4 is stable for most runs.",
                }),
                "lam": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Ridge regularization strength. Higher values (0.5-1.0) prevent artifacts in FP16/FP8 runs.",
                }),
                "window_size": ("INT", {
                    "default": 2, "min": 1, "max": 10,
                    "tooltip": "Initial number of steps to skip between real denoiser evaluations.",
                }),
                "flex_window": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Window growth per real denoiser pass. Higher = more aggressive acceleration, more risk of degradation.",
                }),
                "warmup_steps": ("INT", {
                    "default": 6, "min": 0, "max": 40,
                    "tooltip": "Full denoiser steps before forecasting begins. Ensures stable composition before skipping.",
                }),
                "stop_caching_step": ("INT", {
                    "default": -1, "min": -1, "max": 500, "step": 1,
                    "tooltip": "Step at which Spectrum stops and hands back to the native denoiser. "
                               "Set to (total_steps - 3) for best detail recovery. "
                               "-1 = auto (stops at 80% of steps). 500 = disable guard entirely.",
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 500, "step": 1,
                    "tooltip": "Must match your KSampler total steps. Used to normalize the Chebyshev timestep coordinates.",
                }),
                "calibrated": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Calibrated mode: records residual correction after each real denoiser pass "
                               "and blends it into forecasts, recovering washed-out details.",
                }),
                "calibration_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Residual blending strength in calibrated mode. 0.5-0.8 recommended. Ignored when calibrated=False.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "SwarmUI/sampling"
    DESCRIPTION = (
        "Applies Spectrum diffusion sampling acceleration to any model "
        "(SDXL, Flux, Wan, SD3, HunyuanVideo, Chroma, etc.). "
        "Skips selected denoising steps with Chebyshev polynomial forecasts, "
        "achieving significant speed-ups with minimal quality loss."
    )

    def patch(self, model, w, m, lam, window_size, flex_window,
              warmup_steps, stop_caching_step, steps, calibrated, calibration_strength):
        cal_strength = float(calibration_strength) if calibrated else 0.0
        state = {
            "forecasters": None,
            "cnt": 0,
            "num_cached": [],
            "curr_ws": float(window_size),
            "last_t": float("inf"),
            "total_steps": steps,
        }

        def spectrum_wrapper(model_function, kwargs):
            x = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            batch_size = x.shape[0]
            t_scalar = float(timestep.flatten()[0]) if isinstance(timestep, torch.Tensor) else float(timestep)
            # Detect the start of a new denoising pass (timestep reset to a higher value)
            if t_scalar > state["last_t"]:
                state["forecasters"] = None
                state["cnt"] = 0
                state["num_cached"] = [0] * batch_size
                state["curr_ws"] = float(window_size)
            state["last_t"] = t_scalar
            if state["forecasters"] is None:
                state["forecasters"] = [_ChebyshevForecaster(m=m, lam=lam) for _ in range(batch_size)]
                for f in state["forecasters"]:
                    f.total_steps = steps
            if len(state["num_cached"]) != batch_size:
                state["num_cached"] = [0] * batch_size
            # Decide which batch items run the real denoiser vs. a forecast
            do_real = torch.ones(batch_size, dtype=torch.bool, device=x.device)
            for i in range(batch_size):
                if state["cnt"] < warmup_steps:
                    continue
                if stop_caching_step == -1:
                    if state["cnt"] >= int(steps * 0.8):
                        continue
                elif stop_caching_step < 500 and state["cnt"] >= stop_caching_step:
                    continue
                ws = max(1, math.floor(state["curr_ws"]))
                do_real[i] = (state["num_cached"][i] + 1) % ws == 0
            out = torch.empty_like(x)
            # --- Real denoiser forward pass ---
            if do_real.any():
                real_idx = do_real.nonzero(as_tuple=False).flatten().tolist()
                x_real = x[do_real]
                ts_real = _slice_batch(timestep, real_idx, batch_size)
                c_real = {k: _slice_batch(v, real_idx, batch_size) for k, v in c.items()}
                out_real = model_function(x_real, ts_real, **c_real)
                out[do_real] = out_real
                for j, idx in enumerate(real_idx):
                    f = state["forecasters"][idx]
                    if calibrated:
                        f.record_residual(out_real[j])
                    f.update(state["cnt"], out_real[j])
                    state["num_cached"][idx] = 0
            # --- Forecasted steps ---
            do_forecast = ~do_real
            if do_forecast.any():
                forecast_idx = do_forecast.nonzero(as_tuple=False).flatten().tolist()
                for j, idx in enumerate(forecast_idx):
                    out[idx] = state["forecasters"][idx].predict(state["cnt"], w, calibration_strength=cal_strength)
                    state["num_cached"][idx] += 1
            if state["cnt"] >= warmup_steps:
                state["curr_ws"] += flex_window
            state["cnt"] += 1
            return out

        new_model = model.clone()
        new_model.set_model_unet_function_wrapper(spectrum_wrapper)
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "SwarmSpectrum": SwarmSpectrum,
}
