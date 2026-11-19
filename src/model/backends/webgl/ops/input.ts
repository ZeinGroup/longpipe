import lightingSrc from '../shaders/lighting_input.glsl'
import type { DisplayLighting, LightAdjustment, ImageSource, InputOp } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op.ts'
import inputSrc from '~/model/backends/webgl/shaders/input.glsl'
import { orientationUvMatrix, type Orientation } from '~/model/orientation.ts'

const QUAD_VERT = `#version 300 es
const vec2 VERTS[6] = vec2[6](
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0,1.0),  vec2(1.0,-1.0), vec2(1.0,1.0)
);
void main() { gl_Position = vec4(VERTS[gl_VertexID], 0.0, 1.0); }`

// Standalone (doesn't extend WebGLOp) because the source isn't a Tensor and
// the source texture needs LINEAR filtering rather than the NEAREST that
// WebGLOp.makeTexture defaults to. Builds its own program + source texture +
// output texture, but uses backend.tensor for the output so other ops can
// read it as a regular Tensor.
export class InputWebGL implements InputOp {
  readonly output: WebGLTensor

  private readonly gl: WebGL2RenderingContext
  private readonly raw: ReturnType<InputWebGL['compile']>
  private lit?: ReturnType<InputWebGL['compile']>
  private readonly srcTex: WebGLTexture
  private orientation: Orientation | null = null

  private source: ImageSource | null = null

  constructor(private readonly backend: WebGLBackend, h: number, w: number) {
    const gl = backend.gl
    this.gl = gl
    this.output = backend.tensor(h, w, 4)

    this.srcTex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S,     gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T,     gl.CLAMP_TO_EDGE)

    this.raw = this.compile(inputSrc)
  }

  private compile(source: string) {
    const gl = this.gl
    const vert = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vert, QUAD_VERT)
    gl.compileShader(vert)
    const frag = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(frag, source)
    gl.compileShader(frag)
    if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS))
      throw new Error(`Input GLSL compile error: ${gl.getShaderInfoLog(frag)}`)

    const program = gl.createProgram()!
    gl.attachShader(program, vert)
    gl.attachShader(program, frag)
    gl.linkProgram(program)
    gl.deleteShader(vert)
    gl.deleteShader(frag)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
      throw new Error(`Input GLSL link error: ${gl.getProgramInfoLog(program)}`)
    const loc = (name: string) => gl.getUniformLocation(program, name)
    return { program, src: loc('u_src'), w: loc('u_out_w'), h: loc('u_out_h'),
      uvX: loc('u_uv_x'), uvY: loc('u_uv_y'), gain: loc('u_gain'), guide: loc('u_guide'), half: loc('u_half') }

  }

  // `orientation`: display transform still to apply (see model/orientation.ts);
  // applied while sampling. Omitted = identity.
  setSource(src: ImageSource, orientation?: Orientation | null): void {
    this.source = src
    this.orientation = orientation ?? null
  }

  run(lighting?: DisplayLighting): void {
    if (!this.source) throw new Error('InputWebGL.run() called before setSource()')

    const gl = this.gl
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex)
    // texImage2D accepts ImageBitmap and VideoFrame as TexImageSource.
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
      this.source as TexImageSource,
    )

    const adjustment = lighting?.prepare(out => this.draw(out as WebGLTensor))
    this.draw(this.output, adjustment)
  }

  private draw(out: WebGLTensor, light?: LightAdjustment | null): void {
    const gl = this.gl
    const p = light
      ? this.lit ??= this.compile(inputSrc.replace('// DISPLAY_LIGHTING', '#define DISPLAY_LIGHTING\n' + lightingSrc))
      : this.raw
    gl.useProgram(p.program)
    if (light) {
      gl.activeTexture(gl.TEXTURE1)
      gl.bindTexture(gl.TEXTURE_2D, (light.guide as WebGLTensor).texture)
      gl.uniform1i(p.guide, 1)
      gl.activeTexture(gl.TEXTURE2)
      gl.bindTexture(gl.TEXTURE_2D, (light.half as WebGLTensor).texture)
      gl.uniform1i(p.half, 2)
      gl.uniform4f(p.gain, light.gain, light.localGain, light.red, light.blue)
    }
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex)
    gl.uniform1i(p.src, 0)
    gl.uniform1i(p.w, out.w)
    gl.uniform1i(p.h, out.h)
    const m = orientationUvMatrix(this.orientation)
    gl.uniform3f(p.uvX, m[0], m[1], m[2])
    gl.uniform3f(p.uvY, m[3], m[4], m[5])
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.backend.fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, out.texture, 0)
    gl.viewport(0, 0, out.texW, out.texH)
    gl.bindVertexArray(null)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }
}
