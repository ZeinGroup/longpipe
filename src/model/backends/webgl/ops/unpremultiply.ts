import type { Tensor } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/unpremultiply.glsl'

// Normalized convolution, part 2: rgb / max(a, eps) after blurring (rgb·w, w). Where the
// blur footprint is all subject (a -> 0) it fades to black, which only shows
// where the subject's own alpha is ~1 anyway.
export class UnpremultiplyWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, input: Tensor) {
    super(backend)
    const t = input as WebGLTensor
    this.output = { h: input.h, w: input.w, c: 4, texture: this.makeTexture(null, t.texW, t.texH), texW: t.texW, texH: t.texH }
    this.inputs = [input]
    this.samplers = [
      { name: 'u_input', texture: (input as WebGLTensor).texture },
    ]
    this.defaultSetup()
    this.dispatch = [t.texW, t.texH]
  }
}
