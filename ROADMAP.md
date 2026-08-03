# Chiaroscuro Forge Roadmap

## Phase 5: Color Science Modernization

The project uses CIE 1976 L\*a\*b\* as its working color space, Euclidean a\*b\* distance (CIE76) for color fidelity measurement, and luminance-only SSIM/PSNR for quality assessment. These foundations are technically correct but represent the consensus of the 1990s–2000s. Modern color science, computational photography, and perceptual quality assessment have advanced considerably. The following three items bring the project's color science foundations into alignment with the current state of the art.

### 1. Migrate quality assessment from SSIM/PSNR to CIEDE2000 + LPIPS

**Goal:** Replace outdated perceptual metrics with modern, empirically-validated alternatives that correlate more strongly with human judgments of image quality.

**Why this matters:** The project currently uses:
- SSIM and PSNR (luminance-only, grayscale conversion before computation at `metrics.py:49-52`), which are blind to color shifts and have r ≈ 0.7 correlation with human quality ratings.
- CIE76 ΔE\*ab Euclidean distance in the a\*b\* plane (`metrics.py:431`), deprecated by the CIE in 2001 in favor of CIEDE2000, which corrects for lightness, chroma, and hue non-uniformities.
- HOG feature similarity (`metrics.py:175-189`), a hand-crafted feature from 2005 superseded by CNN-based perceptual features.

**Planned work:**

- Replace the `color_preservation` metric computation with CIEDE2000 (`skimage.color.deltaE_ciede2000(lab1, lab2)`), which accounts for perceptual non-uniformities in the LAB space. CIEDE2000-weighted color differences correct for the fact that ΔE of 1 in the blue region is perceived differently from ΔE of 1 in the gray region.
- Add an optional `lpips` dependency (Zhang et al., CVPR 2018) and compute LPIPS when available, falling back to MS-SSIM when not installed. LPIPS correlates with human ratings at r ≈ 0.9 versus SSIM's r ≈ 0.7.
- Adjust `calculate_quality_score` weights in `metrics.py` to include LPIPS for the photography and art presets, where perceptual fidelity matters most.
- Keep SSIM/PSNR as fallback options for environments where the learned metric cannot be deployed.

**Expected outcome:** Quality scores that meaningfully reflect human perception, enabling reliable automated comparison of enhancement methods and presets.

**References:** CIE 248:2022 (CIECAM16); ISO/CIE 11664-6:2014 (CIEDE2000); Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," CVPR 2018.

### 2. Add HDR-aware linear light processing with a proper tone mapping stage

**Goal:** Process images in physically-correct linear light rather than gamma-encoded sRGB, and add a dedicated tone mapping stage for perceptually-motivated dynamic range compression.

**Why this matters:** The pipeline currently assumes 8-bit sRGB [0, 1] input and applies all operations — Gaussian blur, unsharp masking, ratio-based color preservation — in gamma-encoded space. This is incorrect for physically-based image manipulation:
- Blurring and blending in gamma space amplifies highlights and attenuates shadows because gamma encoding compresses bright tones and expands dark tones.
- The `SharpenStage` (unsharp mask, `pipeline.py:123-126`) computes `image + sharpen_amount * (image - blurred)`, where `blurred` is computed in gamma space. The high-pass component `image - blurred` has a different physical interpretation than it would in linear light.
- The `_preserve_ratio` method (`pipeline.py:247-263`) computes `original / orig_sum` — RGB channel intensity ratios — in gamma space. Intensity ratios are only physically meaningful in linear light.

**Planned work:**

- Add an inverse-gamma (linearization) stage at the pipeline entry: apply the sRGB electro-optical transfer function (EOTF) to convert from gamma-encoded to linear light: `linear = np.where(image <= 0.04045, image/12.92, ((image+0.055)/1.055)**2.4)`.
- Perform all linear operations (denoise, sharpen, LAB-space decomposition) in the linear domain. The LAB conversion already assumes linear sRGB input when using `skimage.color.rgb2lab`, so this change corrects rather than modifies the existing pipeline.
- Add a `ToneMappingStage` that applies a perceptually-motivated global or local tone mapping operator. Start with Reinhard et al.'s global photographic operator (2002): `L_display = L / (1 + L)` for the luminance channel, followed by sRGB encoding.
- Offer a `--linear` flag or `ProcessingConfig(linear_light=True)` that enables this processing path, defaulting to gamma-space for backward compatibility with existing presets and benchmarks.
- This naturally resolves the Gamma/ColorPreservation ordering issue — tone mapping becomes the final perceptual stage after color preservation, and gamma correction operates as a separate tonal sculpting step rather than an encoding artefact.

**Expected outcome:** Physically correct image processing that produces predictable results regardless of input dynamic range, and a foundation for future HDR input support.

**References:** Reinhard et al., "Photographic Tone Reproduction for Digital Images," SIGGRAPH 2002; Mantiuk et al., "Display Adaptive Tone Mapping," SIGGRAPH 2008; IEC 61966-2-1:1999 (sRGB specification).

### 3. Implement gradient-domain local contrast enhancement to realize the "chiaroscuro" concept

**Goal:** Add a spatially-aware contrast enhancement stage that manipulates image gradients rather than global histograms, realizing the local light-and-shadow modelling implied by the project's Renaissance-inspired name.

**Why this matters:** The project's identity — "inspired by Renaissance techniques" — implies local, spatially-aware contrast manipulation that models volume through light and shadow. The current `ContrastStage` applies only global histogram operations (percentile stretch, histogram equalization, CLAHE), which cannot selectively enhance shadow detail while preserving highlights in a scene-appropriate way. CLAHE provides local contrast within fixed-size tiles but does not respect scene semantics or illumination boundaries.

Gradient-domain techniques solve exactly this problem: they manipulate the image's gradient field (the rate of luminance change at each pixel) and then solve a Poisson equation to reconstruct the enhanced image. By attenuating or amplifying gradients based on local scene properties, one can selectively enhance shadow detail, preserve highlight rolloff, and model the kind of volumetric light effects that define Renaissance chiaroscuro.

**Planned work:**

- Add a `ChiaroscuroStage` (`pipeline.py`) that operates after contrast enhancement and before color preservation:
  1. Decompose the image into L\*, a\*, and b\* channels via `color.rgb2lab`.
  2. Compute the L\* channel's gradient field using a Sobel operator: `G_x = filters.sobel_h(l_channel)`, `G_y = filters.sobel_v(l_channel)`.
  3. Apply a spatially-varying attenuation function to gradient magnitudes: amplify gradients in shadow regions (L\* < 50), preserve in midtones, and attenuate in bright regions (L\* > 80). This creates the characteristic chiaroscuro effect — deep, detailed shadows against controlled highlights.
  4. Reconstruct the enhanced L\* channel by solving the Poisson equation: `div(G) = laplacian(l_enhanced)`. For small images, use `scipy.sparse.linalg.cg`; for production, implement a Jacobi or multigrid solver.
  5. Blend the enhanced L\* with the original a\* and b\* channels using the existing `_preserve_lab` method.
- Add a `ProcessingConfig.chiaroscuro_strength: float = 0.0` parameter (0–1) controlling the gradient amplification factor, defaulting to 0 (disabled) for backward compatibility.
- Set `chiaroscuro_strength = 0.4` in the "art" preset (`config.py`) since gradient-domain manipulation is most aesthetically relevant for artwork and photography.
- Add a `--chiaroscuro` flag to the CLI.

**Expected outcome:** A genuinely novel processing stage that gives the project its name. The existing pipeline is a standard enhancement chain; this addition makes it unique. The gradient-domain approach is grounded in the computational photography literature and aligns naturally with the Renaissance metaphor.

**References:** Fattal et al., "Gradient Domain High Dynamic Range Compression," SIGGRAPH 2002; Pérez et al., "Poisson Image Editing," SIGGRAPH 2003; Gastal & Oliveira, "Domain Transform for Edge-Aware Image and Video Processing," SIGGRAPH 2011.

---

## Success criteria

- Quality scores correlate with human perceptual judgments (r > 0.85 on standard benchmarks).
- The "chiaroscuro" name is backed by a genuine gradient-domain algorithm rather than borrowed metaphor.
- Linear-light processing path produces physically correct results for HDR-capable input.
