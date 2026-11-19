import type { Tensor, MLBuffer, BoxFilterParams } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import { WebGPUTensor, WebGPUOp } from '~/model/backends/webgpu/base_webgpu_op.ts'
import f32Src from '~/model/backends/webgpu/shaders/box_filter.wgsl'
import f16Src from '~/model/backends/webgpu/shaders/box_filter_f16.wgsl'

// Guided upsampler, step 2: separable box mean along one axis (see the shader).
// Binding order (inputs -> uniforms -> output).
export class BoxFilterWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGPUTensor
  protected dispatch: [number, number, number]
  shader: string

  constructor(backend: WebGPUBackend, input: Tensor, params: BoxFilterParams) {
    super(backend)
    this.shader = backend.dtype === 'f16' ? f16Src : f32Src
    const n = input.h * input.w
    this.output = backend.tensor(input.h, input.w, 4)
    this.inputs = [input]
    this.createUniform('params', 'Params')
    this.setUniform('params', new Uint32Array([input.w, input.h, params.radius, params.vertical ? 1 : 0]))
    this.defaultSetup()
    this.dispatch = [Math.ceil(input.w / 8), Math.ceil(input.h / 8), 1]
  }
}
