import type { Tensor, MLBuffer } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import { WebGPUTensor, WebGPUOp } from '~/model/backends/webgpu/base_webgpu_op.ts'
import f32Src from '~/model/backends/webgpu/shaders/unpremultiply.wgsl'
import f16Src from '~/model/backends/webgpu/shaders/unpremultiply_f16.wgsl'

// Normalized convolution, part 2: rgb / max(a, eps) after blurring (rgb·w, w). Where the
// blur footprint is all subject (a -> 0) it fades to black, which only shows
// where the subject's own alpha is ~1 anyway.
// Binding order (inputs -> uniforms -> output).
export class UnpremultiplyWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGPUTensor
  protected dispatch: [number, number, number]
  shader: string

  constructor(backend: WebGPUBackend, input: Tensor) {
    super(backend)
    this.shader = backend.dtype === 'f16' ? f16Src : f32Src
    const n = input.h * input.w
    this.output = backend.tensor(input.h, input.w, 4)
    this.inputs = [input]
    this.createUniform('params', 'Params')
    this.setUniform('params', new Uint32Array([n, 0, 0, 0]))
    this.defaultSetup()
    this.dispatch = [Math.ceil(n / 256), 1, 1]
  }
}
