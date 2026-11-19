import type { Tensor, GuidedCoeffsParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/guided_coeffs.glsl'

// Guided upsampler, step 3: a = cov(I,p)/(var(I)+eps), b = mean(p) − a·mean(I).
export class GuidedCoeffsWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, input: Tensor, params: GuidedCoeffsParams) {
    super(backend)
    const t = input as WebGLTensor
    this.output = { h: input.h, w: input.w, c: 4, texture: this.makeTexture(null, input.w, input.h), texW: input.w, texH: input.h }
    this.inputs = [input]
    this.samplers = [
      { name: 'u_input', texture: (input as WebGLTensor).texture },
    ]
    this.uniformFloats = { u_eps: params.eps }
    this.defaultSetup()
    this.dispatch = [t.w, t.h]
  }
}
