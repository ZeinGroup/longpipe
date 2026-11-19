import type { Tensor, MLBuffer, AlphaRampParams } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import { WebGPUTensor, WebGPUOp } from '~/model/backends/webgpu/base_webgpu_op.ts'
import f32Src from '~/model/backends/webgpu/shaders/alpha_ramp.wgsl'
import f16Src from '~/model/backends/webgpu/shaders/alpha_ramp_f16.wgsl'

// Display-alpha ramp: a = smoothstep(lo, hi, alpha.x). Tightens the soft edge the
// compositor blends over, so less of the original (sharp) room shows through the
// transition around a subject. Applied to the display alpha only (after the
// temporal stabilizer, never fed back into it). alpha in .x; output (a, 0, 0, 0).
// Binding order (inputs -> uniforms -> output).
export class AlphaRampWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGPUTensor
  protected dispatch: [number, number, number]
  shader: string

  constructor(backend: WebGPUBackend, input: Tensor, params: AlphaRampParams) {
    super(backend)
    this.shader = backend.dtype === 'f16' ? f16Src : f32Src
    const n = input.h * input.w
    this.output = backend.tensor(input.h, input.w, 4)
    this.inputs = [input]
    this.createUniform('params', 'Params')
    const u = new Uint32Array(4)
    u[0] = n
    const f = new Float32Array(u.buffer)
    f[1] = params.lo
    f[2] = params.hi
    this.setUniform('params', u)
    this.defaultSetup()
    this.dispatch = [Math.ceil(n / 256), 1, 1]
  }
}
