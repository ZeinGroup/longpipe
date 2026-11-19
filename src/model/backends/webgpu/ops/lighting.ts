import type { MLBuffer, Tensor } from '~/model/backend.ts'
import type { WebGPUBackend } from '../index'
import { WebGPUOp, type WebGPUTensor } from '../base_webgpu_op'
import points from '../shaders/lighting_points.wgsl'
import guide from '../shaders/lighting_guide.wgsl'
import meter from '../shaders/lighting_meter.wgsl'

export class LightingWebGPU extends WebGPUOp {
  readonly weights: MLBuffer[] = []
  readonly output: WebGPUTensor

  private constructor(b: WebGPUBackend, readonly inputs: Tensor[], shader: string,
    dims: number[], w: number, h: number, channels: number, protected dispatch: [number, number, number]) {
    super(b)
    this.shader = (b.dtype === 'f16' ? 'enable f16;\nalias Pixel = vec4<f16>;\n' : 'alias Pixel = vec4<f32>;\n') + shader
    this.output = b.tensor(h, w, channels)
    this.createUniform('dims', '')
    this.setUniform('dims', new Uint32Array(dims))
    this.defaultSetup()
  }
  static points(b: WebGPUBackend, hm: Tensor, box: Tensor) {
    return new LightingWebGPU(b, [hm, box], points, [hm.w, hm.h], 5, 1, 4, [1, 1, 1])
  }
  static guide(b: WebGPUBackend, alpha: Tensor, kp: Tensor, w: number, h: number) {
    return new LightingWebGPU(b, [alpha, kp], guide, [alpha.w, alpha.h, w, h], w, h, 4, [Math.ceil(w / 8), Math.ceil(h / 8), 1])
  }
  static meter(b: WebGPUBackend, raw: Tensor, mask: Tensor) {
    const w = raw.w / 6, h = raw.h / 6
    return new LightingWebGPU(b, [raw, mask], meter, [raw.w, raw.h], w, h, 12, [Math.ceil(w / 8), Math.ceil(h / 4), 1])
  }
}
