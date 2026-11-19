# Automatic lighting

Enable automatic lighting when creating the pipeline, or connect it to a checkbox during a call. It defaults to off. The output track stays live across changes.

```ts
const pipeline = new EffectsPipeline(stream, { background: 'none', lighting: true })
await pipeline.ready
await pipeline.setLighting(false)
await pipeline.setLighting(true)
```

Lighting adjusts the camera before touch-up, reframe and background composition. Blur inherits the adjustment. Image, colour and video replacements keep their own colours. The global `setEnabled(false)` switch bypasses lighting too. Both GPU backends and both segmentation model families support the switch.

## How it works

The display input shares its camera upload with the meter and the half-resolution chroma filter. The browser converts camera range, matrix and transfer to the upload's full-range sRGB space. Lighting decodes that RGB to linear light for exposure and white balance, then encodes it back to sRGB. The network and optical flow still sample the original camera. Comparisons involving limited-range YUV, wide-gamut or HDR sources depend on the browser's SDR conversion.

| Fixed setting | Purpose |
| --- | --- |
| 96 × 54 meter, transposed in portrait, every 200 ms | Sample the upright camera independently of frame rate. |
| 6 × 6 sample tiles, discard the top and bottom 15% of weighted tile means | Average in log₂ luminance while limiting the influence of a lamp or dark corner. Values below 0.004 linear luminance and channels at or above 0.975 encoded are excluded. Fewer than 96 valid samples attenuate the lift. |
| 260 weighted person samples, 32 face samples for full confidence | Blend the scene estimate toward the person and then the face as support grows. Lighting reuses an active matte and the five heatmaps available during reframe or touch-up. With lighting alone and no background, it meters the whole frame and leaves inference idle. |
| Cheek and forehead regions in eye-to-eye units, keypoint confidence ramp 0.15–0.30 | Follow face scale and roll, excluding the eyes and mouth. Small, low-confidence or implausibly proportioned detections lose their weight. |
| Subject floor 0.115, scene reference 0.105 linear, 68% compensation up to 1.65 EV | Lift dim midtones while preserving differences between complexions. A bright scene limits the global lift to 0.65 EV. |
| Up to 0.48 EV additional local fill, 25% of the measured deficit | Favour a backlit subject. Inner and outer matte samples at 1.5 and 3 meter cells feather the fill inside the silhouette. The face receives the strongest part of the broad fill. |
| Neutral confidence fades from 0.23 to 0.65 stops of channel imbalance, saturating at 320 weighted samples | Estimate log green/red and green/blue from near-neutral midtones with preference for the background. Apply at most 65% of the measured cast, capped at ±0.22 stops. Weak evidence preserves ambient warmth. |
| Shadow toe 0.002–0.025 linear, peak-channel shoulder and a gamut knee at 0.80 | Protect black and preserve hue through highlight roll-off. Unity exposure and gains use the original ingest shader. |
| Half-resolution bilinear chroma reference, up to 60% chroma blending | Preserve full-resolution luma. The filter fades across luma differences of 0.025–0.09, chroma differences of 0.065–0.16 and brighter midtones. |
| Attack 0.65 s, release 1.8 s, deadbands 0.045 EV and 0.012 log₂ gain | Damp camera auto-exposure and white-balance oscillation. Slew is limited to 0.85 EV/s and 0.12 WB stops/s, with 55% of those rates on release. |
| Cut when over 60% of tiles jump by 1.5 EV or 0.32 in summed red/blue chromaticity distance | Release the previous scene's settings. Camera switches and toggles also reset the controller and discard outstanding results. |

Each meter result contains 144 summaries. WebGL2 queues a pixel-pack buffer transfer and copies from it after a zero-timeout fence poll succeeds. WebGPU maps a staging buffer asynchronously. Only one measurement can be in flight. The render loop keeps displaying frames while it waits.

## Checking cost

The following are **incremental estimates** for a warmed-up 1280 × 720 stream on UHD 630. They need hardware measurement. An already well-lit frame only pays for metering at 5 Hz.

| Work | Estimated GPU time | Basis |
| --- | ---: | --- |
| Five keypoint refinements and interior guide | 0.03–0.16 ms | Five bounded searches on meter ticks. About 5k guide cells on corrected frames with a matte. |
| Raw meter sample, tile reductions and transfer | 0.05–0.25 ms per tick | 5,184 samples and at most 6.75 KiB of summaries. |
| Half-resolution chroma reference | 0.12–0.35 ms | One filtered camera read and one write per 2 × 2 output pixels. |
| Correction fused into display ingest | 0.45–1.20 ms | One cached half-resolution read, four small guide reads and tone arithmetic per pixel. The existing full-resolution write serves all compositors. |

Time lighting on/off in the same pipeline, including meter ticks, with disjoint timer queries on WebGL2 or timestamp queries on WebGPU where available. Separately count frames and frame-age percentiles on the **outgoing track**. Compare no background, blur and replacement with the same source, preset and transport.
