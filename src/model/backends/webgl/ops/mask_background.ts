import type { Tensor } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/mask_background.glsl'

// Background-only blur input (normalized convolution, part 1): the image weighted by the
// background's share w = 1 - alpha, with w itself in .a. Run through the blur
// pyramid and divided back (Unpremultiply), the blurred background takes no
// colour from the subject: no halo of the person around the silhouette.
export class MaskBackgroundWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, image: Tensor, alpha: Tensor) {
    super(backend)
    const t = image as WebGLTensor
    this.output = { h: image.h, w: image.w, c: 4, texture: this.makeTexture(null, t.texW, t.texH), texW: t.texW, texH: t.texH }
    this.inputs = [image, alpha]
    this.samplers = [
      { name: 'u_image', texture: (image as WebGLTensor).texture },
      { name: 'u_alpha', texture: (alpha as WebGLTensor).texture },
    ]
    this.defaultSetup()
    this.dispatch = [t.texW, t.texH]
  }
}
