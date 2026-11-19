import { describe, expect, it, vi } from 'vitest'
import type { Backend } from '~/model/backend'
import { AutoLighting } from '~/model/effects/auto_lighting'
import { LightingController, lightingCurve, meterLevels } from '~/model/effects/lighting'

// The GPU's tile layout. Make each group independently controllable so tests
// distinguish subject metering from an incidental change in the whole scene.
function tiles(scene: number, person = scene, face = person, pw = 0, fw = 0) {
  const t = new Float32Array(144 * 12)
  for (let i = 0; i < t.length; i += 12) {
    t.set([36 * scene, pw * person, fw * face, scene, 36, pw, fw, 0, 0, 0, 0, 0], i)
  }
  return t
}
const unity = { ev: 0, local: 0, red: 0, blue: 0 }

describe('automatic lighting meter', () => {
  it('leaves well-lit and completely rejected samples alone', () => {
    expect(meterLevels(tiles(-2.5))).toEqual(unity)
    const sparse = new Float32Array(12)
    expect(meterLevels(sparse)).toEqual(unity)
    sparse.set([-7, 0, 0, -7, 1])
    expect(meterLevels(sparse).ev).toBeLessThan(0.02)
  })
  it('trims bright outliers and blends by actual face and interior mass', () => {
    const base = tiles(-5)
    const clean = meterLevels(base)
    for (let i = 0; i < 12 * 12; i += 12) base[i] = -0.1 * 36
    expect(meterLevels(base).ev).toBeCloseTo(clean.ev, 6)
    const brightest = meterLevels(tiles(-2))
    const body = meterLevels(tiles(-2, -3.5, -5, 10))
    const tinyFace = meterLevels(tiles(-2, -3.5, -5, 10, 0.01))
    const face = meterLevels(tiles(-2, -3.5, -5, 10, 2))
    expect(body.ev).toBeGreaterThan(brightest.ev)
    expect(tinyFace.ev).toBeGreaterThan(body.ev)
    expect(tinyFace.ev).toBeLessThan(face.ev)
    expect(face.local).toBeGreaterThan(0)
    expect(face.local).toBeLessThanOrEqual(0.48)
  })
  it('bounds white balance, fades weak evidence and preserves it at unity exposure', () => {
    const measure = (scene: number, mass: number) => {
      const t = tiles(scene)
      for (let i = 0; i < t.length; i += 12) t.set([mass, -0.6 * mass, 0.6 * mass], i + 7)
      return meterLevels(t)
    }
    expect(measure(-8, 30)).toEqual({ ev: 1.65, local: 0, red: -0.22, blue: 0.22 })
    expect(Math.abs(measure(-8, 0.01).red)).toBeLessThan(0.005)
    expect(measure(-2, 30).red).toBe(0)
  })
})

describe('linear light tone curve', () => {
  it('is identity at unity including black and saturated highlights', () => {
    for (const rgb of [[0, 0, 0], [0.001, 0.008, 0.01], [0.3, 0.22, 0.12], [1, 0.6, 0.99]])
      lightingCurve(rgb, 1, 1, 1).forEach((v, i) => expect(v).toBeCloseTo(rgb[i], 12))
  })
  it('preserves grey ordering, protects the noise floor and rolls smoothly up to white', () => {
    let prev = -1
    for (let i = 0; i <= 1000; i++) {
      const v = i / 1000
      const [out] = lightingCurve([v, v, v], 3, 1, 1)
      expect(out).toBeGreaterThan(prev)
      expect(out).toBeGreaterThanOrEqual(v - 1e-12)
      expect(out).toBeLessThanOrEqual(1)
      if (v <= 0.002) expect(out).toBe(v)
      prev = out
    }
    const [r, g, b] = lightingCurve([1, 0.8, 0.7], 3, 2 ** 0.22, 1)
    expect(r).toBeLessThanOrEqual(1)
    expect(g / b).toBeCloseTo(0.8 / 0.7, 8)
  })
})

describe('lighting controller', () => {
  it('slews in stops with elapsed time, without overshoot at different frame rates', () => {
    for (const dt of [16, 67, 1000]) {
      const c = new LightingController()
      const t = tiles(-7)
      for (let i = 0; i < t.length; i += 12) t.set([30, 15, -15], i + 7)
      c.observe(t)
      let prev = c.step(0)
      for (let now = dt; now < 5000; now += dt) {
        const v = c.step(now)
        expect(v.ev - prev.ev).toBeGreaterThanOrEqual(0)
        expect(v.ev - prev.ev).toBeLessThanOrEqual(0.85 * dt / 1000 + 1e-10)
        expect(v.red - prev.red).toBeLessThanOrEqual(0.12 * dt / 1000 + 1e-10)
        expect(v.ev).toBeLessThanOrEqual(1.65)
        expect(v.blue).toBeGreaterThanOrEqual(-0.22)
        prev = v
      }
    }
  })
  it('holds a small AE oscillation in the deadband, releases slowly to exact identity', () => {
    const c = new LightingController()
    c.observe(tiles(-4))
    let now = 0
    while (now < 5000) { c.step(now); now += 40 }
    const held = c.step(now)
    c.observe(tiles(-3.99))
    expect(c.step(now + 40)).toEqual(held)
    c.observe(tiles(-3))
    const released = c.step(now + 80)
    expect(held.ev - released.ev).toBeGreaterThan(0)
    expect(held.ev - released.ev).toBeLessThanOrEqual(0.85 * 0.55 * 0.04)
    for (now += 120; now < 30000; now += 40) c.step(now)
    expect(c.step(now)).toEqual(unity)
  })
  it('discards old gains after a camera switch, luminance cut or same-luma colour cut', () => {
    for (const change of ['camera', 'light', 'colour']) {
      const c = new LightingController()
      c.observe(tiles(-6)); c.step(0)
      expect(c.step(200).ev).toBeGreaterThan(0)
      const next = tiles(change === 'light' ? -2 : -6)
      if (change === 'colour') for (let i = 10; i < next.length; i += 12) next[i] = 0.5
      if (change === 'camera') c.reset()
      else c.observe(next)
      expect(c.step(220)).toEqual(unity)
    }
  })
})

it('keeps one meter read in flight and discards a result from the previous camera', async () => {
  let now = 100
  const clock = vi.spyOn(performance, 'now').mockImplementation(() => now)
  let deliver!: (data: Float32Array) => void
  const readbackDeferred = vi.fn(() => new Promise<Float32Array>(resolve => { deliver = resolve }))
  const tensor = (h: number, w: number, c: number) => ({ h, w, c })
  const meter = { run: vi.fn(), output: tensor(9, 16, 12) }
  const backend = { tensor, readbackDeferred, ops: { LightingMeter: () => meter } } as unknown as Backend
  const lighting = new AutoLighting(backend, 1280, 720)
  const sample = vi.fn()
  const settle = () => new Promise(resolve => setTimeout(resolve, 0))
  try {
    expect(lighting.prepare(sample)).toBeNull()
    await settle()
    now += 400
    expect(lighting.prepare(sample)).toBeNull()
    expect(readbackDeferred).toHaveBeenCalledTimes(1)
    lighting.reset()
    lighting.prepare(sample)
    deliver(tiles(-7))
    await settle()
    now += 100
    expect(lighting.prepare(sample)).toBeNull()
    await settle()
    expect(readbackDeferred).toHaveBeenCalledTimes(2)
    deliver(tiles(-7))
    await settle()
    now += 100
    expect(lighting.prepare(sample)?.gain).toBeGreaterThan(1)
    expect(meter.run).toHaveBeenCalledTimes(2)
  } finally { clock.mockRestore() }
})
