import { describe, expect, it, vi } from 'vitest'
import type { Backend } from '~/model/backend'
import { AutoLighting } from '~/model/effects/auto_lighting'
import { GLOBAL_MAX, LOCAL_MAX, LightingController, SUBJECT_TARGET, lightingCurve, meterLevels } from '~/model/effects/lighting'

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
    // A dim room: a person brighter than the room asks for less over the whole
    // picture (no overshoot), a darker person for more fill inside the matte. The face
    // does not move the level (the person's region does); the interior blends in by
    // its mass.
    const room = meterLevels(tiles(-4.25))
    const bright = meterLevels(tiles(-4.25, -3.25, -2.75, 10))
    expect(room.ev).toBeGreaterThan(bright.ev)
    expect(meterLevels(tiles(-4.25, -3.25, -2.75, 10, 2))).toEqual(bright)
    const dark = meterLevels(tiles(-4.25, -5.5, -5.5, 10))
    const thin = meterLevels(tiles(-4.25, -5.5, -5.5, 0.5))
    expect(dark.local).toBeGreaterThanOrEqual(bright.local)
    expect(dark.local).toBeLessThanOrEqual(LOCAL_MAX)
    expect(thin.local).toBeLessThan(dark.local)
  })
  // A reference video-call app in a dim room: subject 0.044 linear, room around 0.016,
  // +1.8 EV over the whole picture and about +0.5 EV more inside the person.
  it('lands the subject near mid grey in a dark room, like the reference', () => {
    const subject = Math.log2(0.044)
    const lv = meterLevels(tiles(Math.log2(0.016), subject, subject, 10, 2))
    expect(lv.ev).toBeGreaterThan(1.7)
    expect(lv.ev).toBeLessThanOrEqual(GLOBAL_MAX)
    expect(lv.local).toBeGreaterThan(0.4)
    expect(lv.local).toBeLessThanOrEqual(LOCAL_MAX)
    const g = 2 ** (lv.ev + lv.local), s = 0.044
    const lifted = g * s / (1 + (g - 1) * s)   // the reference: 0.044 -> 0.186
    expect(lifted).toBeGreaterThan(SUBJECT_TARGET - 0.01)
    expect(lifted).toBeLessThan(0.2)
    // with no matte the whole correction is global, up to the same ceiling
    const flat = meterLevels(tiles(Math.log2(0.016), subject, subject, 10, 2), false)
    expect(flat.local).toBe(0)
    expect(flat.ev).toBeCloseTo(GLOBAL_MAX, 6)
  })
  it('falls smoothly to identity as the room brightens, whatever the face', () => {
    let prev = Infinity
    for (let scene = -7; scene <= -2; scene += 0.05) {
      const lv = meterLevels(tiles(scene, scene + 1, scene + 1, 10, 2))
      const total = lv.ev + lv.local
      expect(total).toBeLessThanOrEqual(prev + 1e-9)
      if (prev !== Infinity) expect(prev - total).toBeLessThan(0.12)
      prev = total
    }
    expect(prev).toBe(0)
    // a darker complexion in a well-lit room is left alone
    expect(meterLevels(tiles(-2.5, -4.5, -4.5, 10, 2))).toEqual(unity)
  })
  it('bounds white balance, fades weak evidence and preserves it at unity exposure', () => {
    const measure = (scene: number, mass: number) => {
      const t = tiles(scene)
      for (let i = 0; i < t.length; i += 12) t.set([mass, -0.6 * mass, 0.6 * mass], i + 7)
      return meterLevels(t)
    }
    expect(measure(-8, 30)).toEqual({ ev: GLOBAL_MAX, local: 0, red: -0.22, blue: 0.22 })
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
      if (v <= 0.0004) expect(out).toBe(v)
      prev = out
    }
    // Above the toe a grey follows the reference's measured map at its gain
    // (g = 3.48): off code -> on code, within 8 levels at each of the 11 points.
    const reference = [[24, 44], [40, 70], [56, 101], [72, 122], [88, 139], [104, 160], [120, 180], [136, 191], [152, 203], [168, 212], [184, 222]]
    const dec = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
    const enc = (y: number) => (y <= 0.0031308 ? 12.92 * y : 1.055 * y ** (1 / 2.4) - 0.055)
    for (const [off, on] of reference) {
      const y = dec(off / 255)
      expect(Math.abs(enc(lightingCurve([y, y, y], 3.48, 1, 1)[1]) * 255 - on)).toBeLessThan(8)
    }
    const [r, g, b] = lightingCurve([1, 0.8, 0.7], 3, 2 ** 0.22, 1)
    expect(r).toBeLessThanOrEqual(1)
    expect(g / b).toBeCloseTo(0.8 / 0.7, 8)
  })
})

describe('lighting controller', () => {
  it('reaches a new target within a second of switching on, without overshoot at different frame rates', () => {
    for (const dt of [16, 67, 1000]) {
      const c = new LightingController()
      const t = tiles(-7)
      for (let i = 0; i < t.length; i += 12) t.set([30, 15, -15], i + 7)
      c.observe(t)
      const target = meterLevels(t)
      let prev = c.step(0)
      for (let now = dt; now < 5000; now += dt) {
        const v = c.step(now)
        expect(v.ev - prev.ev).toBeGreaterThanOrEqual(0)
        expect(v.ev - prev.ev).toBeLessThanOrEqual(4 * dt / 1000 + 1e-10)
        expect(v.red - prev.red).toBeLessThanOrEqual(0.6 * dt / 1000 + 1e-10)
        expect(v.ev).toBeLessThanOrEqual(target.ev + 1e-12)
        expect(v.blue).toBeGreaterThanOrEqual(-0.22)
        if (now >= 1000) expect(v.ev).toBeGreaterThan(0.9 * target.ev)
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
    c.observe(tiles(-2.5))   // the room is lit now: back to identity, slowly
    const released = c.step(now + 80)
    expect(held.ev - released.ev).toBeGreaterThan(0)
    expect(held.ev - released.ev).toBeLessThanOrEqual(0.85 * 0.55 * 0.04)
    for (now += 120; now < 30000; now += 40) c.step(now)
    expect(c.step(now)).toEqual(unity)
  })
  it('discards old gains after a camera switch', () => {
    const c = new LightingController()
    c.observe(tiles(-6)); c.step(0)
    expect(c.step(200).ev).toBeGreaterThan(0)
    c.reset()
    expect(c.step(220)).toEqual(unity)
  })
  it('keeps the gains through a scene cut, then reaches the new scene within a second', () => {
    for (const change of ['light', 'colour']) {
      const c = new LightingController()
      c.observe(tiles(-6))
      let now = 0
      for (; now < 3000; now += 40) c.step(now)
      const before = c.step(now)
      const next = tiles(change === 'light' ? -2 : -6)
      if (change === 'colour') for (let i = 10; i < next.length; i += 12) next[i] = 0.5
      c.observe(next)
      const after = c.step(now + 40)
      expect(Math.abs(after.ev - before.ev)).toBeLessThanOrEqual(4 * 0.04 + 1e-9)   // no one-frame jump
      for (now += 40; now < 4100; now += 40) c.step(now)
      const settled = c.step(now)
      if (change === 'light') expect(settled).toEqual(unity)
      else expect(settled.ev).toBeCloseTo(before.ev, 1)
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
