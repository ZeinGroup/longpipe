import type { Tensor, MLBuffer, GuidedCoeffsParams } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import { WebGPUTensor, WebGPUOp } from '~/model/backends/webgpu/base_webgpu_op.ts'
import f32Src from '~/model/backends/webgpu/shaders/guided_coeffs.wgsl'
import f16Src from '~/model/backends/webgpu/shaders/guided_coeffs_f16.wgsl'

// Guided upsampler, step 3: a = cov(I,p)/(var(I)+eps), b = mean(p) − a·mean(I).
// Binding order (inputs -> uniforms -> output).
export class GuidedCoeffsWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGPUTensor
  protected dispatch: [number, number, number]
  shader: string

  constructor(backend: WebGPUBackend, input: Tensor, params: GuidedCoeffsParams) {
    super(backend)
    this.shader = backend.dtype === 'f16' ? f16Src : f32Src
    const n = input.h * input.w
    this.output = backend.tensor(input.h, input.w, 4)
    this.inputs = [input]
    this.createUniform('params', 'Params')
    const u = new Uint32Array(4)
    u[0] = n
    new Float32Array(u.buffer)[1] = params.eps
    this.setUniform('params', u)
    this.defaultSetup()
    this.dispatch = [Math.ceil(n / 256), 1, 1]
  }
}
