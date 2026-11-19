import type { Tensor, MLBuffer, StabilizeParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/stabilize.glsl'
import chgSrc from '~/model/backends/webgl/shaders/stabilize_chg.glsl'

// Flow-gated stabilizer. All inputs are 4-ch (1 group) at flow resolution.
export class StabilizeWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(
    backend: WebGLBackend, flow: Tensor, pred: Tensor, ref: Tensor,
    envPrev: Tensor, params: StabilizeParams,
  ) {
    super(backend)
    // The change gate lives in its own shader: with it off, this op is upstream's
    // byte for byte (same shader, same uniforms).
    const gate = (params.chgScale ?? 0) > 0
    if (gate) this.shader = chgSrc

    const W = flow.w, H = flow.h
    const outTexture = this.makeTexture(null, W, H)
    this.output = { h: H, w: W, c: 4, texture: outTexture, texW: W, texH: H }
    this.inputs = [flow, pred, ref, envPrev]

    this.samplers = [
      { name: 'u_flow',     texture: (flow    as WebGLTensor).texture },
      { name: 'u_pred',     texture: (pred    as WebGLTensor).texture },
      { name: 'u_ref',      texture: (ref     as WebGLTensor).texture },
      { name: 'u_env_prev', texture: (envPrev as WebGLTensor).texture },
    ]
    this.uniformInts   = { u_w: W, u_h: H, u_step_x: params.stepX, u_step_y: params.stepY }
    this.uniformFloats = {
      u_t_lo: params.tLo, u_t_hi: params.tHi, u_leak: params.leak, u_release: params.release,
      u_t_div: params.tDiv, u_div_scale: params.divScale,
      ...(gate ? { u_t_chg: params.tChg ?? 0, u_chg_scale: params.chgScale! } : {}),
    }

    this.defaultSetup()
    this.dispatch = [W, H]
  }
}
