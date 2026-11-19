import type { Tensor, StabilizeParams, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import f32Src from "~/model/backends/webgpu/shaders/stabilize.wgsl";
import f16Src from "~/model/backends/webgpu/shaders/stabilize_f16.wgsl";
import chgF32Src from "~/model/backends/webgpu/shaders/stabilize_chg.wgsl";
import chgF16Src from "~/model/backends/webgpu/shaders/stabilize_chg_f16.wgsl";

// Flow-gated stabilizer. Binding order (inputs → uniforms → output):
// 0 flow, 1 pred, 2 ref, 3 envPrev, 4 params, 5 output.
export class StabilizeWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(
    backend: WebGPUBackend, flow: Tensor, pred: Tensor, ref: Tensor,
    envPrev: Tensor, params: StabilizeParams,
  ) {
    super(backend);
    // The change gate lives in its own shaders: with it off, this op is upstream's
    // byte for byte (same shader, same 40-byte uniform).
    const gate = (params.chgScale ?? 0) > 0;
    this.shader = gate
      ? (backend.dtype === "f16" ? chgF16Src : chgF32Src)
      : (backend.dtype === "f16" ? f16Src : f32Src);

    this.output = backend.tensor(flow.h, flow.w, 4);
    this.inputs = [flow, pred, ref, envPrev];

    this.createUniform("params", "Params");
    // { h,w:u32, t_lo,t_hi,leak,release,t_div,div_scale:f32, step_x,step_y:u32 [, t_chg,chg_scale:f32] }
    const u = new Uint32Array(gate ? 12 : 10);
    u[0] = flow.h;
    u[1] = flow.w;
    const fv = new Float32Array(u.buffer);
    fv[2] = params.tLo;
    fv[3] = params.tHi;
    fv[4] = params.leak;
    fv[5] = params.release;
    fv[6] = params.tDiv;
    fv[7] = params.divScale;
    u[8]  = params.stepX;
    u[9]  = params.stepY;
    if (gate) {
      fv[10] = params.tChg ?? 0;
      fv[11] = params.chgScale!;
    }
    this.setUniform("params", u);

    this.defaultSetup();
    this.dispatch = [Math.ceil(flow.w / 8), Math.ceil(flow.h / 8), 1];
  }
}
