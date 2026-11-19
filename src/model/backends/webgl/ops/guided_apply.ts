import type { Tensor, GuidedApplyParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/guided_apply.glsl'

// Guided upsampler, step 4: alpha = clamp(a·I + b) at output resolution, optional ramp.
export class GuidedApplyWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, image: Tensor, coeffs: Tensor, params: GuidedApplyParams) {
    super(backend)
    const t = image as WebGLTensor
    this.output = { h: image.h, w: image.w, c: 4, texture: this.makeTexture(null, image.w, image.h), texW: image.w, texH: image.h }
    this.inputs = [image, coeffs]
    this.samplers = [
      { name: 'u_image', texture: (image as WebGLTensor).texture },
      { name: 'u_coeffs', texture: (coeffs as WebGLTensor).texture },
    ]
    this.uniformInts = { u_out_w: image.w, u_out_h: image.h, u_in_w: coeffs.w, u_in_h: coeffs.h }
    this.uniformFloats = { u_lo: params.lo, u_hi: params.hi }
    this.defaultSetup()
    this.dispatch = [t.w, t.h]
  }
}
