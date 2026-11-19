import { describe, it, expect } from 'vitest'
import { BACKENDS_DTYPE } from '../helpers/backends'
import { runBatched } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { lightingCurve } from '~/model/effects/lighting'

// Exercise native textures/buffers, including packed fp16 summaries and the
// PBO read path. A shader failure on a supported backend must fail its row.
const decode = (c: number) => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
const encode = (c: number) => c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055

describe.each(BACKENDS_DTYPE)('automatic lighting ($name)', ({ name, create }) => {
  async function unsupported() {
    return name === 'WebGPU-f16' && !(await WebGPUBackend.hasF16Support())
  }

  it('reads log sums and confidence masses asynchronously, rejecting clipped and noise-floor pixels', async ctx => {
    if (await unsupported()) return ctx.skip()
    const b = await create()
    try {
      const w = 12, h = 6
      const rgb = new Float32Array(w * h * 4), g = new Float32Array(w * h * 4)
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const c = x === 0 ? 0.02 : x === 1 ? 0.99 : x < 6 ? 0.25 : 0.5
        rgb.set([c, c, c, 1], (y * w + x) * 4)
        g.set([0, 0.5, 0.25, 0], (y * w + x) * 4)
      }
      const op = b.ops.LightingMeter(b.tensor(h, w, 4, rgb), b.tensor(h, w, 4, g))
      runBatched(b, () => op.run())
      const t = await (b.readbackDeferred?.(op.output) ?? b.readback(op.output))
      expect(Array.from(t.slice(4, 7))).toEqual([24, 12, 6])
      expect(t[0] / 24).toBeCloseTo(Math.log2(decode(0.25)), 2)
      expect(t[2] / 6).toBeCloseTo(t[0] / 24, 2)
      expect(t[12] / 36).toBeCloseTo(Math.log2(decode(0.5)), 2)
      expect(t[7]).toBeGreaterThan(0)
      expect(t[8]).toBe(0)
    } finally { b.destroy() }
  })

  it('weights cheeks and forehead, excludes eyes and mouth, and erodes a hard matte', async ctx => {
    if (await unsupported()) return ctx.skip()
    const b = await create()
    try {
      const w = 96, h = 54, heat = new Float32Array(w * h * 8), matte = new Float32Array(w * h * 4)
      for (let y = 0; y < h; y++) for (let x = 11; x < w; x++) matte[(y * w + x) * 4] = 1
      const peaks = [[38, 23], [57, 23], [47, 32], [40, 41], [55, 41]]
      peaks.forEach(([x, y], k) => { heat[(y * w + x) * 8 + k] = 1 })
      const box = b.tensor(1, 1, 4, new Float32Array([0.5, 0.5, 0.4, 0.9]))
      const kp = b.ops.LightingPoints(b.tensor(h, w, 8, heat), box)
      const guide = b.ops.LightingGuide(b.tensor(h, w, 4, matte), kp.output, w, h)
      runBatched(b, () => { kp.run(); guide.run() })
      const data = await b.readback(guide.output)
      const at = (x: number, y: number, c: number) => data[(y * w + x) * 4 + c]
      expect(at(10, 28, 0)).toBe(0)
      expect(at(11, 28, 0)).toBe(0)
      expect(at(24, 28, 1)).toBe(1)
      expect(at(90, 50, 0)).toBeCloseTo(0.85, 2)   // the body takes 85% of the fill, the face all of it
      expect(at(38, 33, 2)).toBeGreaterThan(0.8)
      expect(at(47, 14, 2)).toBeGreaterThan(0.8)
      expect(at(38, 23, 2)).toBe(0)
      expect(at(47, 41, 2)).toBe(0)
    } finally { b.destroy() }
  })

  it('attenuates shadow chroma noise and preserves a high-contrast edge', async ctx => {
    if (await unsupported()) return ctx.skip()
    const b = await create()
    const pixels = new Uint8ClampedArray(8 * 8 * 4)
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const rgb = x > 3 ? [210, 210, 210] : (x + y) % 2 ? [64, 60, 49] : [48, 63, 65]
      pixels.set([...rgb, 255], (y * 8 + x) * 4)
    }
    const src = await createImageBitmap(new ImageData(pixels, 8, 8))
    try {
      const input = b.ops.DisplayInput(8, 8)
      input.setSource(src)
      const guide = b.tensor(1, 1, 4)
      const render = async (size: number) => {
        const reference = b.tensor(size, size, 4)
        input.run({ prepare(sample) {
          sample(reference)
          return { half: reference, guide, gain: 2.5, localGain: 1, red: 1, blue: 1 }
        } })
        return b.readback(input.output)
      }
      // A full-size reference equals the original at every pixel (zero residual).
      const plain = await render(8), filtered = await render(4)
      const chromaSwing = (v: Float32Array) => Math.abs((v[0] - v[2]) - (v[4] - v[6]))
      expect(chromaSwing(filtered)).toBeLessThan(0.8 * chromaSwing(plain))
      expect(filtered[4 * 4]).toBeCloseTo(plain[4 * 4], 3)
    } finally { src.close(); b.destroy() }
  })

  it('smooths flat shadow luma at a strong gain, keeps the edge and the brights', async ctx => {
    if (await unsupported()) return ctx.skip()
    const b = await create()
    const pixels = new Uint8ClampedArray(8 * 8 * 4)
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const v = x > 3 ? 200 : (x + y) % 2 ? 46 : 34
      pixels.set([v, v, v, 255], (y * 8 + x) * 4)
    }
    const src = await createImageBitmap(new ImageData(pixels, 8, 8))
    try {
      const input = b.ops.DisplayInput(8, 8)
      input.setSource(src)
      const guide = b.tensor(1, 1, 4)
      const render = async (size: number) => {
        const reference = b.tensor(size, size, 4)
        input.run({ prepare(sample) {
          sample(reference)
          return { half: reference, guide, gain: 3.48, localGain: 1, red: 1, blue: 1 }
        } })
        return b.readback(input.output)
      }
      // A full-size reference equals the original at every pixel: nothing to smooth.
      const plain = await render(8), filtered = await render(4)
      const at = (v: Float32Array, x: number, y: number) => v[(y * 8 + x) * 4 + 1]
      expect(Math.abs(at(filtered, 1, 1) - at(filtered, 2, 1))).toBeLessThan(0.6 * Math.abs(at(plain, 1, 1) - at(plain, 2, 1)))
      expect(at(filtered, 4, 2)).toBeCloseTo(at(plain, 4, 2), 3)   // the bright side of the edge
      expect(at(filtered, 6, 3)).toBeCloseTo(at(plain, 6, 3), 3)
    } finally { src.close(); b.destroy() }
  })

  // Webcam colour blotches (4:2:0 chroma, 8x8-block compression) at a strong lift: the
  // colour of a flat dark area comes from the 1/16-resolution average. Luma, the mean
  // colour and a neighbouring area of another colour and level stay as they were.
  it('evens out colour blotches in flat darks at a strong lift, keeps luma, the mean colour and the edge', async ctx => {
    if (await unsupported()) return ctx.skip()
    const b = await create()
    const A = [46, 38, 40], B = [34, 41, 42], C = [200, 180, 150]   // A and B: same luma, different colour
    const pixels = new Uint8ClampedArray(32 * 32 * 4)
    for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++)
      pixels.set([...(x >= 16 ? C : ((x >> 3) + (y >> 3)) % 2 ? B : A), 255], (y * 32 + x) * 4)
    const src = await createImageBitmap(new ImageData(pixels, 32, 32))
    try {
      const input = b.ops.DisplayInput(32, 32)
      input.setSource(src)
      const guide = b.tensor(1, 1, 4), half = b.tensor(16, 16, 4)
      const quarter = b.ops.BilinearUpsample(half, { outH: 8, outW: 8 })
      const eighth = b.ops.BilinearUpsample(quarter.output, { outH: 4, outW: 4 })
      const sixteenth = b.ops.BilinearUpsample(eighth.output, { outH: 2, outW: 2 })
      const render = async (wide: boolean) => {
        runBatched(b, () => input.run({ prepare(sample) {
          sample(half); quarter.run(); eighth.run(); sixteenth.run()
          return { half, wide: wide ? sixteenth.output : undefined, guide, gain: 4.9, localGain: 1, red: 1, blue: 1 }
        } }))
        return b.readback(input.output)
      }
      const before = await render(false), after = await render(true)
      const px = (v: Float32Array, x: number, y: number) => Array.from(v.slice((y * 32 + x) * 4, (y * 32 + x) * 4 + 3))
      const luma = (c: number[]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
      const swing = (v: Float32Array) => Math.abs((px(v, 3, 3)[0] - px(v, 3, 3)[2]) - (px(v, 3, 11)[0] - px(v, 3, 11)[2]))
      expect(swing(after)).toBeLessThan(0.5 * swing(before))
      for (const [x, y] of [[3, 3], [3, 11]]) expect(Math.abs(luma(px(after, x, y)) - luma(px(before, x, y)))).toBeLessThan(3 / 255)
      const meanRB = (v: Float32Array) => { let s = 0; for (let y = 0; y < 16; y++) for (let x = 0; x < 8; x++) { const c = px(v, x, y); s += c[0] - c[2] } return s / 128 }
      expect(Math.abs(meanRB(after) - meanRB(before))).toBeLessThan(0.15 * Math.abs(meanRB(before)) + 2 / 255)
      for (const [x, y] of [[17, 5], [20, 20], [16, 12]]) px(after, x, y).forEach((c, i) => expect(c).toBeCloseTo(px(before, x, y)[i], 3))
    } finally { src.close(); b.destroy() }
  })

  it('normalizes limited and full-range YUV before the linear lift', async ctx => {
    if (await unsupported()) return ctx.skip()
    const b = await create()
    const input = b.ops.DisplayInput(8, 8), half = b.tensor(4, 4, 4), guide = b.tensor(1, 1, 4)
    try {
      const levels: number[] = []
      for (const fullRange of [true, false]) {
        const yuv = new Uint8Array(96).fill(128)
        yuv.fill(fullRange ? 64 : 71, 0, 64)
        const frame = new VideoFrame(yuv, {
          format: 'I420', codedWidth: 8, codedHeight: 8,
          timestamp: 0, colorSpace: { primaries: 'bt709', transfer: 'iec61966-2-1', matrix: 'bt709', fullRange }
        })
        try {
          input.setSource(frame)
          runBatched(b, () => input.run({
            prepare(sample) {
              sample(half)
              return { half, guide, gain: 2, localGain: 1, red: 1, blue: 1 }
            }
          }))
          levels.push((await b.readback(input.output))[0])
        } finally { frame.close() }
      }
      const y = decode(64 / 255)
      const expected = encode(lightingCurve([y, y, y], 2, 1, 1)[0])
      for (const value of levels) expect(Math.abs(value - expected)).toBeLessThan(3 / 255)
      expect(Math.abs(levels[0] - levels[1])).toBeLessThan(2 / 255)
    } finally { b.destroy() }
  })

  // With lighting off the display Input must be the plain Input's pass (WebKit, which
  // nobody here can test, runs exactly what npm ships): same output, bit for bit,
  // through a resample and a turn, for an ImageBitmap and a VideoFrame.
  it.for(['bitmap', 'frame'])('with lighting off, the display input is the plain input pass (%s)', async (kind, ctx) => {
    if (await unsupported()) return ctx.skip()
    const b = await create()
    const pixels = new Uint8ClampedArray(16 * 8 * 4)
    for (let i = 0; i < 16 * 8; i++) pixels.set([(i * 37) % 256, (i * 11) % 256, (i * 73) % 256, 255], i * 4)
    const bmp = await createImageBitmap(new ImageData(pixels, 16, 8))
    const src = kind === 'frame' ? new VideoFrame(bmp, { timestamp: 0 }) : bmp
    try {
      for (const orientation of [null, { rotation: 90, flip: true }] as const) {
        const plain = b.ops.Input(6, 12), display = b.ops.DisplayInput(6, 12)
        plain.setSource(src, orientation as never); display.setSource(src, orientation as never)
        runBatched(b, () => { plain.run(); display.run() })
        expect(Array.from(await b.readback(display.output))).toEqual(Array.from(await b.readback(plain.output)))
      }
    } finally {
      if (src !== bmp) src.close()
      bmp.close()
      b.destroy()
    }
  })

  it.for(['bitmap', 'frame'])('keeps raw samples independent of the fused %s display correction', async (kind, ctx) => {
    if (await unsupported()) return ctx.skip()
    const b = await create()
    const pixels = new Uint8ClampedArray(8 * 8 * 4)
    for (let i = 0; i < pixels.length; i += 4) pixels.set([51, 44, 37, 255], i)
    const bmp = await createImageBitmap(new ImageData(pixels, 8, 8))
    const src = kind === 'frame' ? new VideoFrame(bmp, { timestamp: 0 }) : bmp
    try {
      const input = b.ops.DisplayInput(8, 8), raw = b.tensor(4, 4, 4)
      const guide = b.tensor(1, 1, 4, new Float32Array([1, 1, 1, 0]))
      input.setSource(src)
      runBatched(b, () => input.run({
        prepare(sample) {
          sample(raw)
          return { half: raw, guide, gain: 2, localGain: 1.2, red: 0.95, blue: 1.08 }
        }
      }))
      const source = await b.readback(raw)
      const output = await b.readback(input.output)
      const linear = Array.from(source.slice(0, 3)).map(decode)
      const want = lightingCurve(linear, 2.4, 0.95, 1.08).map(encode)
      for (let c = 0; c < 3; c++) {
        expect(source[c]).toBeCloseTo(pixels[c] / 255, 2)
        expect(output[c]).toBeCloseTo(want[c], 2)
      }
      input.run()
      const off = await b.readback(input.output)
      expect(Array.from(off.slice(0, 4))).toEqual(Array.from(source.slice(0, 4)))
    } finally {
      if (src !== bmp) src.close()
      bmp.close()
      b.destroy()
    }
  })
})
