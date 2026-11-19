import type { Tensor } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/guided_pack.glsl'

// Guided upsampler, step 1: (I, p, I·p, I·I) at the model's grid (see the shader).
export class GuidedPackWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, guide: Tensor, alpha: Tensor) {
    super(backend)
    const t = guide as WebGLTensor
    this.output = { h: guide.h, w: guide.w, c: 4, texture: this.makeTexture(null, guide.w, guide.h), texW: guide.w, texH: guide.h }
    this.inputs = [guide, alpha]
    this.samplers = [
      { name: 'u_guide', texture: (guide as WebGLTensor).texture },
      { name: 'u_alpha', texture: (alpha as WebGLTensor).texture },
    ]
    this.defaultSetup()
    this.dispatch = [t.w, t.h]
  }
}
