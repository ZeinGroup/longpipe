import type { Tensor, MLBuffer } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import { WebGPUTensor, WebGPUOp } from '~/model/backends/webgpu/base_webgpu_op.ts'
import f32Src from '~/model/backends/webgpu/shaders/guided_pack.wgsl'
import f16Src from '~/model/backends/webgpu/shaders/guided_pack_f16.wgsl'

// Guided upsampler, step 1: (I, p, I·p, I·I) at the model's grid (see the shader).
// Binding order (inputs -> uniforms -> output).
export class GuidedPackWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGPUTensor
  protected dispatch: [number, number, number]
  shader: string

  constructor(backend: WebGPUBackend, guide: Tensor, alpha: Tensor) {
    super(backend)
    this.shader = backend.dtype === 'f16' ? f16Src : f32Src
    const n = guide.h * guide.w
    this.output = backend.tensor(guide.h, guide.w, 4)
    this.inputs = [guide, alpha]
    this.createUniform('params', 'Params')
    this.setUniform('params', new Uint32Array([n, 0, 0, 0]))
    this.defaultSetup()
    this.dispatch = [Math.ceil(n / 256), 1, 1]
  }
}
