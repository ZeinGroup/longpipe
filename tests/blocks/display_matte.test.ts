import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'

// Display-matte ops (AlphaRamp, MaskBackground + Unpremultiply, the guided
// upsampler) and the stabilizer's optional change gate, against JS references.

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
const smooth = (a: number, lo: number, hi: number) => { const t = clamp((a - lo) / (hi - lo), 0, 1); return t * t * (3 - 2 * t) }
const LUMA = [0.299, 0.587, 0.114]

function rgba(h: number, w: number, f: (x: number, y: number) => [number, number, number, number]): Float32Array {
  const a = new Float32Array(h * w * 4)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) a.set(f(x, y), (y * w + x) * 4)
  return a
}

describe.each(BACKENDS)('display matte ($name)', ({ create }) => {
  it('AlphaRamp is smoothstep(lo, hi) on .x', async () => {
    const backend = await create()
    const H = 2, W = 8
    const v = (x: number, y: number) => (y * W + x) / (H * W - 1)
    const t = backend.tensor(H, W, 4, rgba(H, W, (x, y) => [v(x, y), 0, 0, 0]))
    const op = backend.ops.AlphaRamp(t, { lo: 0.3, hi: 0.8 })
    op.run()
    const got = await backend.readback(op.output)
    backend.destroy()
    let err = 0
    for (let p = 0; p < H * W; p++) err = Math.max(err, Math.abs(got[p * 4] - smooth(p / (H * W - 1), 0.3, 0.8)))
    expect(err).toBeLessThan(1e-4)
  })

  it('MaskBackground weights by 1 - alpha and Unpremultiply divides it back', async () => {
    const backend = await create()
    const H = 2, W = 4
    const col = (x: number, y: number): [number, number, number, number] => [0.1 + 0.2 * x, 0.9 - 0.1 * x, 0.3 + 0.4 * y, 1]
    const al = (x: number, y: number) => [0, 0.25, 0.5, 0.99][x] * (y ? 0.5 : 1)
    const image = backend.tensor(H, W, 4, rgba(H, W, col))
    const alpha = backend.tensor(H, W, 4, rgba(H, W, (x, y) => [al(x, y), 0, 0, 0]))
    const mask = backend.ops.MaskBackground(image, alpha)
    const unmask = backend.ops.Unpremultiply(mask.output)
    mask.run(); unmask.run()
    const m = await backend.readback(mask.output)
    const u = await backend.readback(unmask.output)
    backend.destroy()
    let err = 0
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const p = (y * W + x) * 4, w = 1 - al(x, y), c = col(x, y)
      for (let k = 0; k < 3; k++) {
        err = Math.max(err, Math.abs(m[p + k] - c[k] * w), Math.abs(u[p + k] - c[k]))
      }
      err = Math.max(err, Math.abs(m[p + 3] - w))
    }
    expect(err).toBeLessThan(2e-3)
  })

  it('guided upsampler keeps a flat matte flat and snaps a soft edge to the guide edge', async () => {
    const backend = await create()
    // Low-res grid 8x16, output 16x32. Guide: a sharp luminance step at x = 16 (output),
    // i.e. between low-res columns 7 and 8. Matte: the same step, but soft (a ramp
    // over low-res columns 6..9).
    const h = 8, w = 16, Hh = 16, Wh = 32
    const guideLo = backend.tensor(h, w, 4, rgba(h, w, (x) => x < 8 ? [0.2, 0.2, 0.2, 1] : [0.8, 0.8, 0.8, 1]))
    const image = backend.tensor(Hh, Wh, 4, rgba(Hh, Wh, (x) => x < 16 ? [0.2, 0.2, 0.2, 1] : [0.8, 0.8, 0.8, 1]))
    const run = async (matte: (x: number) => number) => {
      const alpha = backend.tensor(h, w, 4, rgba(h, w, (x) => [matte(x), 0, 0, 0]))
      const pack = backend.ops.GuidedPack(guideLo, alpha)
      const b1 = backend.ops.BoxFilter(pack.output, { radius: 2, vertical: false })
      const b2 = backend.ops.BoxFilter(b1.output, { radius: 2, vertical: true })
      const co = backend.ops.GuidedCoeffs(b2.output, { eps: 1e-3 })
      const c1 = backend.ops.BoxFilter(co.output, { radius: 2, vertical: false })
      const c2 = backend.ops.BoxFilter(c1.output, { radius: 2, vertical: true })
      const ap = backend.ops.GuidedApply(image, c2.output, { lo: 0, hi: 0 })
      for (const s of [pack, b1, b2, co, c1, c2, ap]) s.run()
      return backend.readback(ap.output)
    }
    const flat = await run(() => 0.6)
    const soft = await run((x) => clamp((x - 5.5) / 4, 0, 1))
    backend.destroy()
    let flatErr = 0
    for (let p = 0; p < Hh * Wh; p++) flatErr = Math.max(flatErr, Math.abs(flat[p * 4] - 0.6))
    expect(flatErr).toBeLessThan(2e-3)
    // Across the edge (row 8), 2 px either side of the guide's step: a plain bilinear
    // enlargement of the soft matte differs by 0.375 there; the guided one is steeper.
    const row = 8, at = (x: number) => soft[(row * Wh + x) * 4]
    expect(at(17) - at(14)).toBeGreaterThan(0.45)
    expect(at(2)).toBeLessThan(0.1)
    expect(at(30)).toBeGreaterThan(0.9)
  })

  it('Stabilize change gate trusts the fresh matte where it disagrees with the history', async () => {
    const backend = await create()
    const H = 1, W = 4
    const zeros = () => backend.tensor(H, W, 4, new Float32Array(H * W * 4))
    const pred = backend.tensor(H, W, 4, rgba(H, W, (x) => [[1, 0.6, 0.3, 0.1][x], 0, 0, 0]))
    const ref = zeros()
    const base = { tLo: 1, tHi: 3, leak: 0.15, release: 0.9, tDiv: 10, divScale: 1, stepX: 1, stepY: 1 }
    const off = backend.ops.Stabilize(zeros(), pred, ref, zeros(), base)
    const on  = backend.ops.Stabilize(zeros(), pred, ref, zeros(), { ...base, tChg: 0.2, chgScale: 0.3 })
    off.run(); on.run()
    const a = await backend.readback(off.output), b = await backend.readback(on.output)
    backend.destroy()
    for (let x = 0; x < W; x++) {
      const d = [1, 0.6, 0.3, 0.1][x]
      expect(Math.abs(a[x * 4] - 0.15 * d)).toBeLessThan(1e-3)                                    // off: leak only
      const g = Math.max(clamp((d - 0.2) / 0.3, 0, 1), 0.15)
      expect(Math.abs(b[x * 4] - g * d)).toBeLessThan(1e-3)                                       // on: gate opens with |pred - ref|
    }
  })
})
