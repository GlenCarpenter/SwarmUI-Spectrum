using System.IO;
using Newtonsoft.Json.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

// NOTE: Namespace must NOT contain "SwarmUI" (reserved for built-ins).
namespace SpectrumForecaster;

/// <summary>Extension that integrates Spectrum sampling acceleration into SwarmUI via a built-in ComfyUI node.
/// Supports SDXL, Flux, Wan, SD3, HunyuanVideo, Chroma, and any other ComfyUI-compatible diffusion model.
/// Based on the CVPR 2026 paper 'Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration'
/// by Han et al. (Stanford / ByteDance). https://arxiv.org/abs/2603.01623</summary>
public class SpectrumExtension : Extension
{
    /// <summary>Blending weight between local Taylor prediction and global Chebyshev forecast.</summary>
    public static T2IRegisteredParam<double> SpectrumW;

    /// <summary>Number of Chebyshev polynomial basis functions.</summary>
    public static T2IRegisteredParam<int> SpectrumM;

    /// <summary>Ridge regularization strength (lambda).</summary>
    public static T2IRegisteredParam<double> SpectrumLambda;

    /// <summary>Initial number of denoising steps to skip between real denoiser evaluations.</summary>
    public static T2IRegisteredParam<int> SpectrumWindowSize;

    /// <summary>Window size growth increment applied after each real denoiser pass.</summary>
    public static T2IRegisteredParam<double> SpectrumFlexWindow;

    /// <summary>Number of initial full-denoiser steps before forecasting begins.</summary>
    public static T2IRegisteredParam<int> SpectrumWarmupSteps;

    /// <summary>Step at which Spectrum stops accelerating. -1 = auto (80%), 500 = disabled.</summary>
    public static T2IRegisteredParam<int> SpectrumStopCachingStep;

    /// <summary>Whether to enable calibrated mode (residual correction for detail recovery).</summary>
    public static T2IRegisteredParam<bool> SpectrumCalibrated;

    /// <summary>Residual blending strength in calibrated mode.</summary>
    public static T2IRegisteredParam<double> SpectrumCalibrationStrength;

    /// <summary>Parameter group for all Spectrum settings.</summary>
    public static T2IParamGroup SpectrumGroup;

    /// <inheritdoc/>
    public override void OnPreInit()
    {
        // Register our bundled ComfyUI node so it is available on self-start ComfyUI backends.
        string comfyNodesPath = Path.GetFullPath(FilePath + "/ComfyNodes");
        if (!ComfyUISelfStartBackend.CustomNodePaths.Contains(comfyNodesPath))
        {
            ComfyUISelfStartBackend.CustomNodePaths.Add(comfyNodesPath);
        }
    }

    /// <inheritdoc/>
    public override void OnInit()
    {
        // Map the node name so the "comfyui" feature is used for visibility gating.
        // Since SwarmSpectrum ships with this extension, it is always available on self-start backends.
        // For remote ComfyUI backends, users must copy the ComfyNodes folder to their custom_nodes directory.
        ComfyUIBackendExtension.NodeToFeatureMap["SwarmSpectrum"] = "comfyui";
        SpectrumGroup = new("Spectrum", Toggles: true, Open: false, IsAdvanced: true,
            Description: "Applies <b>Spectrum</b> training-free sampling acceleration to any diffusion model "
                + "(SDXL, Flux, Wan, SD3, HunyuanVideo, Chroma, and more).\n"
                + "Spectrum uses Chebyshev polynomial forecasting to skip redundant denoiser evaluations, "
                + "achieving up to ~2-5× speed-up with minimal quality loss.\n"
                + "Paper: <a href=\"https://arxiv.org/abs/2603.01623\">arxiv.org/abs/2603.01623</a> (CVPR 2026)\n"
                + "Recommended starting point for 20-step image generation: "
                + "W=0.30, M=3, Lambda=0.1, Window=2, Flex=0.25, Warmup=6, Stop=-1.");
        SpectrumW = T2IParamTypes.Register<double>(new("[Spectrum] W",
            "[Spectrum]\nBlending weight between local Taylor prediction (0) and global Chebyshev spectral forecast (1).\n"
            + "Lower values (0.3-0.4) favor sharpness via local momentum.\n"
            + "Higher values (0.6-1.0) rely more on global spectral smoothing.\n"
            + "Setting to 0 uses pure Taylor approximation (disables global Chebyshev).",
            "0.30", Min: 0.0, Max: 1.0, Step: 0.05,
            ViewType: ParamViewType.SLIDER,
            Group: SpectrumGroup, FeatureFlag: "comfyui", OrderPriority: 1,
            Examples: ["0.00", "0.30", "0.60", "1.00"]
            ));
        SpectrumM = T2IParamTypes.Register<int>(new("[Spectrum] M",
            "[Spectrum]\nNumber of Chebyshev polynomial basis functions (forecast complexity).\n"
            + "Lower values (3) are more stable for shorter runs. Higher values (5-8) may improve accuracy in longer runs.",
            "3", Min: 1, Max: 8,
            Group: SpectrumGroup, FeatureFlag: "comfyui", OrderPriority: 2,
            Examples: ["3", "4", "6"]
            ));
        SpectrumLambda = T2IParamTypes.Register<double>(new("[Spectrum] Lambda",
            "[Spectrum]\nRidge regression regularization strength.\n"
            + "Higher values (0.5-1.0) prevent latent explosion, rainbow artifacts, and black outputs in FP16/FP8 modes.\n"
            + "Lower values (0.05-0.1) are fine for FP32 or stable runs.",
            "0.1", Min: 0.0, Max: 2.0, Step: 0.05,
            ViewType: ParamViewType.SLIDER,
            Group: SpectrumGroup, FeatureFlag: "comfyui", OrderPriority: 3,
            Examples: ["0.05", "0.1", "0.5", "1.0"]
            ));
        SpectrumWindowSize = T2IParamTypes.Register<int>(new("[Spectrum] Window Size",
            "[Spectrum]\nInitial number of denoising steps to skip between real denoiser evaluations.\n"
            + "Larger values produce more aggressive skipping from the start.",
            "2", Min: 1, Max: 10,
            Group: SpectrumGroup, FeatureFlag: "comfyui", OrderPriority: 4,
            Examples: ["1", "2", "3"]
            ));
        SpectrumFlexWindow = T2IParamTypes.Register<double>(new("[Spectrum] Flex Window",
            "[Spectrum]\nWindow size growth added after each real denoiser pass.\n"
            + "Higher values push acceleration further as generation progresses.\n"
            + "Lower values keep acceleration conservative and reduce structural degradation risk.",
            "0.25", Min: 0.0, Max: 3.0, Step: 0.05,
            ViewType: ParamViewType.SLIDER,
            Group: SpectrumGroup, FeatureFlag: "comfyui", OrderPriority: 5,
            Examples: ["0.0", "0.25", "0.75", "1.5"]
            ));
        SpectrumWarmupSteps = T2IParamTypes.Register<int>(new("[Spectrum] Warmup Steps",
            "[Spectrum]\nFull denoiser steps to run before forecasting begins.\n"
            + "Ensures the model establishes composition before any skipping occurs.\n"
            + "DiT models (Flux, SD3, Wan, HunyuanVideo) may need more warmup (8-12).",
            "6", Min: 0, Max: 40,
            Group: SpectrumGroup, FeatureFlag: "comfyui", OrderPriority: 6,
            Examples: ["4", "6", "8", "10"]
            ));
        SpectrumStopCachingStep = T2IParamTypes.Register<int>(new("[Spectrum] Stop Caching Step",
            "[Spectrum]\nThe step index at which Spectrum stops forecasting and runs only the real denoiser for remaining steps.\n"
            + "Essential for recovering fine micro-details (skin texture, eyes, etc.).\n"
            + "Set to (total steps - 3) for best quality. -1 = auto-stop at 80% of steps. 500 = disabled.",
            "-1", Min: -1, Max: 500,
            Group: SpectrumGroup, FeatureFlag: "comfyui", OrderPriority: 7,
            Examples: ["-1", "17", "22", "27"]
            ));
        SpectrumCalibrated = T2IParamTypes.Register<bool>(new("[Spectrum] Calibrated",
            "[Spectrum]\nEnable calibrated mode: after each real denoiser pass, Spectrum records the difference "
            + "between its forecast and the true output (residual), then blends it into future forecasts.\n"
            + "Recovers washed-out details that standard Spectrum may lose. Recommended for highest quality.",
            "false",
            Group: SpectrumGroup, FeatureFlag: "comfyui", OrderPriority: 8
            ));
        SpectrumCalibrationStrength = T2IParamTypes.Register<double>(new("[Spectrum] Calibration Strength",
            "[Spectrum]\nStrength of residual correction in calibrated mode.\n"
            + "0.5 blends half the correction into forecasts; 0.8 applies more aggressive recovery.\n"
            + "Only active when Calibrated is enabled.",
            "0.5", Min: 0.0, Max: 1.0, Step: 0.05,
            ViewType: ParamViewType.SLIDER,
            Group: SpectrumGroup, FeatureFlag: "comfyui", OrderPriority: 9,
            DependNonDefault: SpectrumCalibrated.Type.ID,
            Examples: ["0.5", "0.8"]
            ));
        // Run as a model gen step at -3, matching TeaCache's priority.
        // This wraps g.LoadingModel after LoRAs, FreeU, and other model patches have been applied.
        WorkflowGenerator.AddModelGenStep(g =>
        {
            if (!g.UserInput.TryGet(SpectrumW, out double w))
            {
                return;
            }
            // Only apply to the base model pass, not the refiner pass.
            if (g.LoadingModelType != "Base")
            {
                return;
            }
            int steps = g.UserInput.Get(T2IParamTypes.Steps, 20);
            string spectrumNode = g.CreateNode("SwarmSpectrum", new JObject()
            {
                ["model"] = g.LoadingModel,
                ["w"] = w,
                ["m"] = g.UserInput.Get(SpectrumM, 3),
                ["lam"] = g.UserInput.Get(SpectrumLambda, 0.1),
                ["window_size"] = g.UserInput.Get(SpectrumWindowSize, 2),
                ["flex_window"] = g.UserInput.Get(SpectrumFlexWindow, 0.25),
                ["warmup_steps"] = g.UserInput.Get(SpectrumWarmupSteps, 6),
                ["stop_caching_step"] = g.UserInput.Get(SpectrumStopCachingStep, -1),
                ["steps"] = steps,
                ["calibrated"] = g.UserInput.Get(SpectrumCalibrated, false),
                ["calibration_strength"] = g.UserInput.Get(SpectrumCalibrationStrength, 0.5)
            });
            g.LoadingModel = [spectrumNode, 0];
        }, -3);
    }
}
