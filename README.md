# SwarmUI Spectrum Extension

A [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) extension that integrates **Spectrum**,  a training-free diffusion sampling acceleration technique directly, into SwarmUI's's generation pipeline.

Spectrum reduces the number of expensive neural network denoiser calls during a sampling run by forecasting the outputs of skipped steps using **Chebyshev polynomial regression**. Depending on model and settings, this can reduce generation time by **2–4×** with minimal quality loss.

---

## What Spectrum Does

A standard 20-step diffusion run calls the denoiser (UNet or DiT) 20 times — each call is typically the most expensive part of generation.

Spectrum intercepts those calls and, after an initial warmup phase, starts **forecasting** a subset of steps instead of computing them. Forecasted outputs are derived from a sliding window of recent true denoiser outputs using:

- **Chebyshev spectral regression** — fits a global polynomial to the trajectory of feature vectors across time
- **Taylor extrapolation** — uses local finite differences for sharp, momentum-based prediction
- A blended combination of the two, controlled by the `W` parameter

The diffusion trajectory is unchanged — all N steps still run. You simply pay GPU compute for fewer of them.

**Optional: Calibrated Mode**  
After each real denoiser pass, Spectrum records the difference between its forecast and the true output (the residual). It then blends this correction into future forecasts to recover washed-out texture and fine details.

---

## Supported Models

Any model compatible with ComfyUI's `set_model_unet_function_wrapper` API:

- **Flux** (up to ~4.79× speedup reported in paper)
- **Wan 2.1** (up to ~4.67× speedup on Wan 2.1-14B)
- **SDXL**
- **SD3 / SD3.5**
- **HunyuanVideo**
- **Chroma**
- Any other ComfyUI-compatible diffusion model

---

## Installation

1. Open a terminal and navigate to your SwarmUI `src/Extensions` folder:
   ```
   cd SwarmUI/src/Extensions
   ```
2. Clone the repository:
   ```
   git clone https://github.com/GlenCarpenter/SwarmUI-Spectrum.git
   ```
3. Restart SwarmUI

## Parameters

All parameters appear in the **Spectrum** group in the generation parameters panel. The group is collapsed by default and requires **Show Advanced Parameters** to be enabled.

Enable the **Spectrum** toggle to activate acceleration.

| Parameter | Default | Description |
|---|---|---|
| **W** | 0.30 | Blend between local Taylor prediction (0) and global Chebyshev forecast (1). Lower values favor sharpness. |
| **M** | 3 | Number of Chebyshev polynomial basis functions. Higher values allow more complex trajectory fitting. |
| **Lambda** | 0.1 | Ridge regularization strength. Increase to 0.5–1.0 when using FP16/FP8 precision to prevent artifacts. |
| **Window Size** | 2 | Initial number of steps to skip between real denoiser evaluations. |
| **Flex Window** | 0.25 | Growth increment added to window size after each real denoiser pass (progressive acceleration). |
| **Warmup Steps** | 6 | Full denoiser evaluations to run before any forecasting begins. DiT models may need 8–12. |
| **Stop Caching Step** | -1 | Step index at which forecasting stops and all remaining steps run the real denoiser. `-1` = auto (80% of steps). `500` = disabled. |
| **Calibrated** | false | Enable residual correction to recover detail that pure forecasting may lose. |
| **Calibration Strength** | 0.5 | How much of the residual correction is blended into forecasts (calibrated mode only). |

### Recommended Starting Points

| Model Family | W | M | Lambda | Window | Warmup | Notes |
|---|---|---|---|---|---|---|
| SDXL | 0.30 | 3 | 0.1 | 2 | 6 | Stable at defaults |
| Flux | 0.30 | 3 | 0.3 | 2 | 8 | More warmup for DiT structure |
| SD3 / SD3.5 | 0.30 | 3 | 0.2 | 2 | 8 | |
| Wan 2.1 | 0.30 | 3 | 0.3 | 2 | 10 | Long video runs benefit from Calibrated mode |
| HunyuanVideo | 0.30 | 4 | 0.5 | 2 | 10 | Higher Lambda for stability |

---

## Tuning Tips

- **Artifacts / blurring / color drift?** Reduce `Window Size` to 1, increase `Lambda`, increase `Warmup Steps`, or lower `Stop Caching Step`.
- **FP16 / FP8 model unstable?** Raise `Lambda` to 0.5–1.0.
- **Want more speed?** Increase `Window Size` and `Flex Window`. Accept that quality loss will increase.
- **Want highest quality?** Enable `Calibrated` mode. Set `Stop Caching Step` to around `(total steps - 3)`.
- **DiT models (Flux, SD3, Wan, HunyuanVideo)** generally need more `Warmup Steps` (8–12) than UNet models.

---

## Attribution

**Spectrum algorithm:**
> Han, J., Zhu, K., Zhang, R., Yang, C., Wu, C., Shen, Y., & Yu, F. (2026).  
> *Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration.*  
> CVPR 2026. [https://arxiv.org/abs/2603.01623](https://arxiv.org/abs/2603.01623)  
> Stanford University & ByteDance. GitHub: [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum)

**SwarmUI:**
> Alex "mcmonkey" Goodwin.  
> [https://github.com/mcmonkeyprojects/SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI)

**SwarmUI extension and ComfyUI node implementation:**  
Written with GitHub Copilot (Claude Sonnet 4.6).

---

## License

MIT
