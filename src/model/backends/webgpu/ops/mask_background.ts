import type { Tensor, MLBuffer } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import { WebGPUTensor, WebGPUOp } from '~/model/backends/webgpu/base_webgpu_op.ts'
import f32Src from '~/model/backends/webgpu/shaders/mask_background.wgsl'
import f16Src from '~/model/backends/webgpu/shaders/mask_background_f16.wgsl'

// Background-only blur input (normalized convolution, part 1): the image weighted by the
// background's share w = 1 - alpha, with w itself in .a. Run through the blur
// pyramid and divided back (Unpremultiply), the blurred background takes no
// colour from the subject: no halo of the person around the silhouette.
// Binding order (inputs -> uniforms -> output).
export class MaskBackgroundWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGPUTensor
  protected dispatch: [number, number, number]
  shader: string

  constructor(backend: WebGPUBackend, image: Tensor, alpha: Tensor) {
    super(backend)
    this.shader = backend.dtype === 'f16' ? f16Src : f32Src
    const n = image.h * image.w
    this.output = backend.tensor(image.h, image.w, 4)
    this.inputs = [image, alpha]
    this.createUniform('params', 'Params')
    this.setUniform('params', new Uint32Array([n, 0, 0, 0]))
    this.defaultSetup()
    this.dispatch = [Math.ceil(n / 256), 1, 1]
  }
}
