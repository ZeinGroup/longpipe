import lightingSrc from "../shaders/lighting_input.wgsl";
import type { DisplayLighting, DisplayInputOp, LightAdjustment, ImageSource, Dtype } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import type { WebGPUTensor } from "~/model/backends/webgpu/base_webgpu_op.ts";
import input2dF32Src        from "~/model/backends/webgpu/shaders/input_2d.wgsl";
import input2dF16Src        from "~/model/backends/webgpu/shaders/input_2d_f16.wgsl";
import inputExternalF32Src  from "~/model/backends/webgpu/shaders/input_external.wgsl";
import inputExternalF16Src  from "~/model/backends/webgpu/shaders/input_external_f16.wgsl";
import { orientationUvMatrix, sameOrientation, type Orientation } from "~/model/orientation.ts";
import { InputWebGPU } from "~/model/backends/webgpu/ops/input.ts";

const IS_VIDEO_FRAME = (s: ImageSource): s is VideoFrame =>
  typeof VideoFrame !== "undefined" && s instanceof VideoFrame;

// The display Input with automatic lighting. With no lighting hook, run() is
// the upstream Input op's own run (same pipelines, import, uniforms, dispatch):
// the lit path below exists only while lighting is on. It shares the upstream
// op's output tensor and builds its shaders from the upstream WGSL text (the
// anchors are checked, so a change upstream fails loudly).
function inject(src: string, anchor: string, text: string): string {
  if (!src.includes(anchor)) throw new Error(`display input: anchor '${anchor}' not found in the input shader`);
  return src.replace(anchor, text);
}

export class DisplayInputWebGPU implements DisplayInputOp {
  readonly output: WebGPUTensor;
  private readonly base: InputWebGPU;

  private readonly backend: WebGPUBackend;
  private readonly device: GPUDevice;
  private readonly dtype: Dtype;
  private readonly sampler: GPUSampler;
  private readonly params = new WeakMap<WebGPUTensor, { buffer: GPUBuffer; orientation: Orientation | null }>();
  private lightParams?: GPUBuffer;
  private lit2d?: GPUComputePipeline;
  private litExternal?: GPUComputePipeline;

  // Lazily-compiled pipelines and the staging texture for the 2D path.
  private pipeline2d: GPUComputePipeline | null = null;
  private pipelineExternal:  GPUComputePipeline | null = null;
  private stagingTex:        GPUTexture | null = null;
  private stagingW = 0;
  private stagingH = 0;

  private source: ImageSource | null = null;
  // Display transform for the next camera sample (null = identity).
  private orientation: Orientation | null = null;

  constructor(backend: WebGPUBackend, h: number, w: number) {
    this.backend = backend;
    this.device = backend.device;
    this.dtype  = backend.dtype;
    this.base   = new InputWebGPU(backend, h, w);
    this.output = this.base.output as WebGPUTensor;

    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
  }

  // `orientation`: the display transform still to apply to `src` (rotation /
  // flip the browser's upload path won't apply — see model/orientation.ts).
  // Applied in the sampling shader; omitted = identity.
  setSource(src: ImageSource, orientation?: Orientation | null): void {
    this.source = src;
    this.orientation = orientation ?? null;
    this.base.setSource(src, orientation);
  }

  // The uniform write is queue-timeline, ahead of the (batched) dispatch that
  // reads it, and only happens when the orientation changes.
  private paramsFor(out: WebGPUTensor): GPUBuffer {
    const entry = this.params.get(out);
    if (entry && sameOrientation(entry.orientation, this.orientation)) return entry.buffer;
    const buffer = entry?.buffer ?? this.device.createBuffer({
      size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const ab = new ArrayBuffer(48);
    new Uint32Array(ab, 0, 2).set([out.w, out.h]);
    const m = orientationUvMatrix(this.orientation);
    new Float32Array(ab, 16, 8).set([m[0], m[1], m[2], 0, m[3], m[4], m[5], 0]);
    this.device.queue.writeBuffer(buffer, 0, ab);
    this.params.set(out, { buffer, orientation: this.orientation });
    return buffer;
  }

  private externalUnsupported = false;

  run(lighting?: DisplayLighting): void {
    if (!lighting) { this.base.run(); return; }   // lighting off: upstream's pass
    const src = this.source;
    if (!src) throw new Error("DisplayInputWebGPU.run() called before setSource()");
    let external: GPUExternalTexture | undefined;
    if (IS_VIDEO_FRAME(src) && !this.externalUnsupported) {
      try {
        external = this.device.importExternalTexture({ source: src, colorSpace: "srgb" });
      } catch (err) {
        this.externalUnsupported = true;
        console.warn("[longpipe] external texture unavailable; using copy path:", err);
      }
    }
    let resource: GPUExternalTexture | GPUTextureView;
    if (external) {
      resource = external;
    } else {
      const w = IS_VIDEO_FRAME(src) ? src.displayWidth : src.width;
      const h = IS_VIDEO_FRAME(src) ? src.displayHeight : src.height;
      this.ensureStagingTexture(w, h);
      this.backend.flush();
      this.device.queue.copyExternalImageToTexture(
        { source: src, flipY: false }, { texture: this.stagingTex!, colorSpace: "srgb" }, [w, h],
      );
      resource = this.stagingTex!.createView();
    }
    const sample = (out: WebGPUTensor, light?: LightAdjustment | null) => {
      const pipeline = this.pipelineFor(!!external, !!light);
      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource }, { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.paramsFor(out) } },
        { binding: 3, resource: { buffer: out.buffer } },
      ];
      if (light) {
        this.lightParams ??= this.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const wide = light.wide ?? light.half;
        const ab = new ArrayBuffer(48);
        new Float32Array(ab, 0, 4).set([light.gain, light.localGain, light.red, light.blue]);
        new Uint32Array(ab, 16).set([light.guide.w,light.guide.h,light.half.w,light.half.h, wide.w, wide.h, 0, 0]);
        this.device.queue.writeBuffer(this.lightParams, 0, ab);
        entries.push({ binding: 4, resource: { buffer: this.lightParams } },
          { binding: 5, resource: { buffer: (light.guide as WebGPUTensor).buffer } },
          { binding: 6, resource: { buffer: (light.half as WebGPUTensor).buffer } },
          { binding: 7, resource: { buffer: (wide as WebGPUTensor).buffer } });
      }
      const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      this.dispatchOnce(pipeline, group, [Math.ceil(out.w/8),Math.ceil(out.h/8),1]);
    };
    const light = lighting?.prepare(out => sample(out as WebGPUTensor));
    sample(this.output, light);
  }

  private pipelineFor(external: boolean, lit: boolean): GPUComputePipeline {
    const source = external
      ? (this.dtype === "f16" ? inputExternalF16Src : inputExternalF32Src)
      : (this.dtype === "f16" ? input2dF16Src : input2dF32Src);
    if (!lit) {
      if (!external) return this.pipeline2d ??= this.buildPipeline(source);
      return this.pipelineExternal ??= this.buildPipeline(source);
    }
    const key = external ? "litExternal" : "lit2d";
    const sampled = external ? "let rgba = textureSampleBaseClampToEdge(" : "let rgba = textureSampleLevel(";
    const litSrc = inject(inject(source, sampled, "var rgba = " + sampled.slice("let rgba = ".length)),
      "    output_buf[y * params.out_w + x] = ",
      "    rgba = vec4<f32>(correctLighting(rgba.rgb, out_uv.xy), rgba.a);\n    output_buf[y * params.out_w + x] = ");
    return this[key] ??= this.buildPipeline(litSrc + `\nalias Pixel = vec4<${this.dtype}>;\n` + lightingSrc);
  }

  private ensureStagingTexture(w: number, h: number): void {
    if (this.stagingTex && this.stagingW === w && this.stagingH === h) return;
    this.stagingTex?.destroy();
    this.stagingTex = this.device.createTexture({
      size: [w, h, 1],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.stagingW = w;
    this.stagingH = h;
  }

  private buildPipeline(code: string): GPUComputePipeline {
    const module = this.device.createShaderModule({ code });
    return this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  private dispatchOnce(pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, dispatch: [number,number,number]): void {
    const shared = this.backend.computePass();   // inside backend.batch()
    if (shared) {
      shared.setPipeline(pipeline);
      shared.setBindGroup(0, bindGroup);
      shared.dispatchWorkgroups(...dispatch);
      return;
    }
    const enc  = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...dispatch);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
