// Exposure and white balance are measured in stops. The meter returns three
// vec4s per 6×6 tile: weighted log sums, masses and neutral chromaticity sums.
// Tile trimming keeps a lamp or a dark lock of hair from driving exposure.
export const METER_W = 96
export const METER_H = 54
export const METER_INTERVAL = 200
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))
const mix = (a: number, b: number, w: number) => a + w * (b - a)

export interface LightLevels {
  ev: number
  local: number
  red: number
  blue: number
}
const identity = (): LightLevels => ({ ev: 0, local: 0, red: 0, blue: 0 })

function trimmedMean(tiles: Float32Array, group: number, fallback: number) {
  const values: { value: number; weight: number }[] = []
  let mass = 0
  for (let i = 0; i < tiles.length; i += 12) {
    const weight = tiles[i + 4 + group]
    if (weight <= 0) continue
    values.push({ value: tiles[i + group] / weight, weight })
    mass += weight
  }
  if (!mass) return { mean: fallback, mass }
  values.sort((a, b) => a.value - b.value)
  let offset = 0, sum = 0
  for (const { value, weight } of values) {
    const kept = Math.max(0, Math.min(offset + weight, mass * 0.85) - Math.max(offset, mass * 0.15))
    sum += kept * value
    offset += weight
  }
  return { mean: sum / (mass * 0.7), mass }
}

export function meterLevels(tiles: Float32Array): LightLevels {
  const scene = trimmedMean(tiles, 0, -3)
  const person = trimmedMean(tiles, 1, scene.mean)
  const face = trimmedMean(tiles, 2, person.mean)
  const personWeight = clamp(person.mass / 260, 0, 1)
  const faceWeight = clamp(face.mass / 32, 0, 1)
  const subject = mix(mix(scene.mean, person.mean, personWeight), face.mean, faceWeight)
  // A lower bound on useful midtones, with partial compensation: preserve
  // differences between skin tones. Let a backlit subject override a bright room.
  const need = Math.max(0, Math.log2(0.115) - subject)
  const sceneNeed = Math.max(0, Math.log2(0.105) - scene.mean)
  const evidence = clamp(scene.mass / 96, 0, 1)
  const ev = evidence * Math.min(1.65, 0.68 * need, 0.65 + 0.75 * sceneNeed)
  const local = evidence * Math.min(0.48, 0.25 * need) * personWeight
  if (ev === 0 && local === 0) return identity()
  let neutral = 0, red = 0, blue = 0
  for (let i = 0; i < tiles.length; i += 12) {
    neutral += tiles[i + 7]
    red += tiles[i + 8]
    blue += tiles[i + 9]
  }
  const strength = clamp(neutral / 320, 0, 1) * Math.min(1, ev * 2) * 0.65
  const gain = (sum: number) => clamp(strength * sum / Math.max(1, neutral), -0.22, 0.22)
  return { ev, local, red: gain(red), blue: gain(blue) }
}

export class LightingController {
  private target = identity()
  private value = identity()
  private last: number | undefined
  private previous: Float32Array | undefined

  reset(): void {
    this.target = identity()
    this.value = identity()
    this.previous = undefined
    this.last = undefined
  }

  observe(tiles: Float32Array): void {
    if (!tiles.every(Number.isFinite)) return
    // Compare spatial log luminance including rejected black/white pixels.
    // An isolated hand crossing the lens should not reset the controller.
    let changed = 0
    if (this.previous) {
      for (let i = 0; i < tiles.length; i += 12) {
        const logJump = Math.abs(tiles[i + 3] - this.previous[i + 3])
        const colourJump = Math.abs(tiles[i + 10] - this.previous[i + 10])
          + Math.abs(tiles[i + 11] - this.previous[i + 11])
        if (logJump > 1.5 || colourJump > 0.32) changed++
      }
      if (changed > 0.6 * tiles.length / 12) this.reset()
    }
    this.previous = tiles
    this.target = meterLevels(tiles)
  }

  step(now: number): LightLevels {
    const dt = Math.max(0, (now - (this.last ?? now)) / 1000)
    this.last = now
    const follow = (current: number, target: number, wb: boolean) => {
      const delta = target - current
      const band = wb ? 0.012 : 0.045
      if (target !== 0 && Math.abs(delta) < band) return current
      const release = Math.abs(target) < Math.abs(current)
      const slew = (wb ? 0.12 : 0.85) * (release ? 0.55 : 1)
      const fraction = target === 0 && Math.abs(delta) < band ? 1 : -Math.expm1(-dt / (release ? 1.8 : 0.65))
      const step = Math.min(Math.abs(delta) * fraction, slew * dt)
      return current + Math.sign(delta) * step
    }
    const v = this.value, t = this.target
    this.value = {
      ev: follow(v.ev, t.ev, false), local: follow(v.local, t.local, false),
      red: follow(v.red, t.red, true), blue: follow(v.blue, t.blue, true)
    }
    return this.value
  }
}

// CPU reference for the two display shaders. Browser ingest produces
// full-range sRGB. Preserve black and apply the shoulder to the brightest
// channel so highly saturated colours also approach white level smoothly.
export function lightingCurve(linear: number[], gain: number, red: number, blue: number): number[] {
  const y = linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722
  const toe = clamp((y - 0.002) / 0.023, 0, 1)
  const lift = 1 + (gain - 1) * toe * toe * (3 - 2 * toe)
  const rgb = [linear[0] * red, linear[1], linear[2] * blue]
  const peak = Math.max(...rgb)
  const scale = lift / (1 + (lift - 1) * peak)
  // A slope-matched knee maps the maximum possible WB excursion to 1.
  const p = peak * scale
  const excess = Math.max(0, p - 0.8)
  const limit = Math.max(1, red, blue)
  const curvature = (limit - 1) / (0.2 * (limit - 0.8))
  const mapped = 0.8 + excess / (1 + curvature * excess)
  const compress = p > 0.8 ? mapped / p : 1
  return rgb.map(c => c * scale * compress)
}
