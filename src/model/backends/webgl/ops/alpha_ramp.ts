import type { Tensor, AlphaRampParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/alpha_ramp.glsl'

// Display-alpha ramp: a = smoothstep(lo, hi, alpha.x). Tightens the soft edge the
// compositor blends over, so less of the original (sharp) room shows through the
// transition around a subject. Applied to the display alpha only (after the
// temporal stabilizer, never fed back into it). alpha in .x; output (a, 0, 0, 0).
export class AlphaRampWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, input: Tensor, params: AlphaRampParams) {
    super(backend)
    const t = input as WebGLTensor
    this.output = { h: input.h, w: input.w, c: 4, texture: this.makeTexture(null, t.texW, t.texH), texW: t.texW, texH: t.texH }
    this.inputs = [input]
    this.samplers = [
      { name: 'u_input', texture: (input as WebGLTensor).texture },
    ]
    this.uniformFloats = { u_lo: params.lo, u_hi: params.hi }
    this.defaultSetup()
    this.dispatch = [t.texW, t.texH]
  }
}
