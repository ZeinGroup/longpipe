import type { Backend, Tensor, Presenter, RenderTarget, Op } from '~/model/backend.ts'
import { BlurPyramid } from '~/model/effects/blur_pyramid.ts'

// Background-blur composite: blurs the input image with a pyramid (output
// stops at half input resolution) and composites it as the bg via the
// bilinear-sampling presenter. The presenter's final per-pixel scan
// expands the half-res blurred bg back to canvas resolution for free —
// avoiding the most expensive op a full pyramid would have.
// Below this sigma there's effectively no blur to do — and the pyramid floors
// at 2 levels (a visible half-res blur) so it can't represent "off" anyway.
// At/under this, composite the subject over the UNMODIFIED background (strength
// 0 = no blur).
const BLUR_EPS = 0.05

export class CompositorBlur {
  private readonly blur:      BlurPyramid | null
  private readonly presenter: Presenter
  // masked: (rgb·(1−a), 1−a) → pyramid → rgb/a, the background blurred without the subject.
  private readonly mask:      Op | null = null
  private readonly unmask:    Op | null = null

  constructor(backend: Backend, image: Tensor, alpha: Tensor, sigma: number, target: RenderTarget = 'main', masked = false) {
    if (sigma <= BLUR_EPS) {
      this.blur      = null
      this.presenter = backend.presenters.CompositeImage(image, alpha, image, target)
    } else if (masked) {
      // Normalized convolution: blur the background weighted by its own share and
      // divide the weight back out, so the blurred room near the silhouette is
      // the room, not the subject smeared outward (published background-blur
      // pipelines weight their blur by the mask for the same reason).
      this.mask      = backend.ops.MaskBackground(image, alpha)
      this.blur      = new BlurPyramid(backend, this.mask.output, sigma)
      this.unmask    = backend.ops.Unpremultiply(this.blur.output)
      this.presenter = backend.presenters.CompositeImageBilinear(image, alpha, this.unmask.output, target)
    } else {
      this.blur      = new BlurPyramid(backend, image, sigma)
      this.presenter = backend.presenters.CompositeImageBilinear(image, alpha, this.blur.output, target)
    }
  }

  run(): void {
    this.mask?.run()
    this.blur?.run()
    this.unmask?.run()
    this.presenter.run()
  }
}
