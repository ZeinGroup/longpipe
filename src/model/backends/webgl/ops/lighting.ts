import type { MLBuffer, Tensor } from '~/model/backend.ts'
import type { WebGLBackend } from '../index'
import { WebGLOp, type WebGLTensor } from '../base_webgl_op'
import points from '../shaders/lighting_points.glsl'
import guide from '../shaders/lighting_guide.glsl'
import meter from '../shaders/lighting_meter.glsl'

export class LightingWebGL extends WebGLOp {
  readonly weights: MLBuffer[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]

  private constructor(backend: WebGLBackend, readonly inputs: Tensor[], names: string[],
    shader: string, w: number, h: number, channels = 4) {
    super(backend)
    this.shader = shader
    this.output = backend.tensor(h, w, channels)
    this.dispatch = [this.output.texW, h]
    this.samplers = inputs.map((t, i) => ({ name: names[i], texture: (t as WebGLTensor).texture }))
    this.uniformInts = { u_w: w, u_h: h }
    this.defaultSetup()
  }
  static points(b: WebGLBackend, hm: Tensor, box: Tensor) {
    return new LightingWebGL(b, [hm, box], ['u_hm', 'u_box'], points, 5, 1)
  }
  static guide(b: WebGLBackend, alpha: Tensor, kp: Tensor, w: number, h: number) {
    return new LightingWebGL(b, [alpha, kp], ['u_alpha', 'u_points'], guide, w, h)
  }
  static meter(b: WebGLBackend, raw: Tensor, mask: Tensor) {
    return new LightingWebGL(b, [raw, mask], ['u_raw', 'u_guide'], meter, raw.w / 6, raw.h / 6, 12)
  }
}
