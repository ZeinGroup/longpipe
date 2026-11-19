import type { Tensor, MLBuffer, GuidedApplyParams } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import { WebGPUTensor, WebGPUOp } from '~/model/backends/webgpu/base_webgpu_op.ts'
import f32Src from '~/model/backends/webgpu/shaders/guided_apply.wgsl'
import f16Src from '~/model/backends/webgpu/shaders/guided_apply_f16.wgsl'

// Guided upsampler, step 4: alpha = clamp(a·I + b) at output resolution, optional ramp.
// Binding order (inputs -> uniforms -> output).
export class GuidedApplyWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGPUTensor
  protected dispatch: [number, number, number]
  shader: string

  constructor(backend: WebGPUBackend, image: Tensor, coeffs: Tensor, params: GuidedApplyParams) {
    super(backend)
    this.shader = backend.dtype === 'f16' ? f16Src : f32Src
    const n = image.h * image.w
    this.output = backend.tensor(image.h, image.w, 4)
    this.inputs = [image, coeffs]
    this.createUniform('params', 'Params')
    const u = new Uint32Array(8)
    u[0] = image.w; u[1] = image.h; u[2] = coeffs.w; u[3] = coeffs.h
    const f = new Float32Array(u.buffer)
    f[4] = params.lo; f[5] = params.hi
    this.setUniform('params', u)
    this.defaultSetup()
    this.dispatch = [Math.ceil(image.w / 8), Math.ceil(image.h / 8), 1]
  }
}
