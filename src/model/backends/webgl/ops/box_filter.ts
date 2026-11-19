import type { Tensor, BoxFilterParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/box_filter.glsl'

// Guided upsampler, step 2: separable box mean along one axis (see the shader).
export class BoxFilterWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, input: Tensor, params: BoxFilterParams) {
    super(backend)
    const t = input as WebGLTensor
    this.output = { h: input.h, w: input.w, c: 4, texture: this.makeTexture(null, input.w, input.h), texW: input.w, texH: input.h }
    this.inputs = [input]
    this.samplers = [
      { name: 'u_input', texture: (input as WebGLTensor).texture },
    ]
    this.uniformInts = { u_w: input.w, u_h: input.h, u_radius: params.radius, u_vertical: params.vertical ? 1 : 0 }
    this.defaultSetup()
    this.dispatch = [t.w, t.h]
  }
}
