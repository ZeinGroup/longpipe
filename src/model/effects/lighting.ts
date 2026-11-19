// Exposure and white balance are measured in stops. The meter returns three
// vec4s per 6×6 tile: weighted log sums, masses and neutral chromaticity sums.
// Tile trimming keeps a lamp or a dark lock of hair from driving exposure.
export const METER_W = 96
export const METER_H = 54
export const METER_INTERVAL = 200
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))
const mix = (a: number, b: number, w: number) => a + w * (b - a)
const smoothstep01 = (t: number) => t * t * (3 - 2 * t)

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

// Fitted to a reference video-call app's lighting adjustment in a dim room with dark
// walls: one rational curve over the whole picture with g = 3.48
// (+1.8 EV at the bottom, white stays white), and +0.4 to +0.65 EV more inside the
// person, who ends near mid grey (0.186 linear).
export const SUBJECT_TARGET = 0.18   // linear luminance of the subject after the curve
// The fill inside the person aims its interior reading a little higher: on the night
// scene the meter's interior reading sits ~20% above the region's median and the body
// takes 65% of the fill (the face all of it), so this lands the region at ~0.18-0.19,
// +0.3 to +0.5 EV above the global curve, as the reference does.
export const LOCAL_TARGET = 0.24
export const GLOBAL_MAX = 1.8        // stops, the whole picture
export const LOCAL_MAX = 0.5         // stops, inside the person only
// How dark the room is decides how much of that response applies, on the meter's own
// scene reading (which skips near-black pixels: the dim room reads 0.045): all of it at
// 0.05 or below, none at 0.14 or above (the well-lit clip reads 0.24 and stays exactly
// as it is), smoothly in between. The face decides only within that envelope, so a
// darker complexion in a well-lit room is left alone rather than pushed toward grey.
const ROOM_DARK = Math.log2(0.05)
const ROOM_LIT = Math.log2(0.14)

// Stops a log2 level needs so that out = G·s / (1 + (G − 1)·s) reaches the target.
export function subjectStops(subject: number, target = SUBJECT_TARGET): number {
  const s = 2 ** subject
  if (s >= target) return 0
  return Math.log2(target * (1 - s) / (s * (1 - target)))
}

// localFill: whether a matte will carry the local part (with no matte the whole
// correction is global and the subject is the scene).
//   global: what the whole picture can take, at most the room's share of GLOBAL_MAX and
//           never more than the scene or the person needs (no overshoot on a subject
//           brighter than its room);
//   local:  what the person's interior still needs to reach LOCAL_TARGET on top, up to
//           the room's share of LOCAL_MAX.
// The subject is the person's interior, as the reference's response was measured on
// the region where the person is. (The face, usually the brightest part of it, read
// high: on the night scene a face-weighted subject left the person at code 103
// instead of ~119.)
export function meterLevels(tiles: Float32Array, localFill = true): LightLevels {
  const scene = trimmedMean(tiles, 0, -3)
  const person = trimmedMean(tiles, 1, scene.mean)
  const personWeight = localFill ? clamp(person.mass / 260, 0, 1) : 0
  const subject = mix(scene.mean, person.mean, personWeight)
  const room = smoothstep01(clamp((ROOM_LIT - scene.mean) / (ROOM_LIT - ROOM_DARK), 0, 1))
  const evidence = clamp(scene.mass / 96, 0, 1)
  const needed = subjectStops(subject)
  const globalRaw = Math.min(GLOBAL_MAX * room, subjectStops(scene.mean), needed)
  const localRaw = Math.min(LOCAL_MAX * room, Math.max(0, subjectStops(subject, LOCAL_TARGET) - globalRaw)) * personWeight
  const ev = evidence * globalRaw
  const local = evidence * localRaw
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
  // Acquiring: right after a switch-on, a camera switch or a scene change, reach the
  // new target within about a second, as the reference does; then track it calmly.
  private acquiring = true
  private seen = false

  reset(): void {
    this.target = identity()
    this.value = identity()
    this.previous = undefined
    this.last = undefined
    this.acquiring = true
    this.seen = false
  }

  observe(tiles: Float32Array, localFill = true): void {
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
      // A new scene: keep the present gains (no one-frame jump to identity) and
      // move to the new scene's quickly.
      if (changed > 0.6 * tiles.length / 12) this.acquiring = true
    }
    this.previous = tiles
    this.target = meterLevels(tiles, localFill)
    this.seen = true
  }

  step(now: number): LightLevels {
    const dt = Math.max(0, (now - (this.last ?? now)) / 1000)
    this.last = now
    const fast = this.acquiring && this.seen
    const follow = (current: number, target: number, wb: boolean) => {
      const delta = target - current
      const band = wb ? 0.012 : 0.045
      if (target !== 0 && Math.abs(delta) < band) return current
      const release = Math.abs(target) < Math.abs(current)
      const slew = fast ? (wb ? 0.6 : 4) : (wb ? 0.12 : 0.85) * (release ? 0.55 : 1)
      const tau = fast ? 0.25 : release ? 1.8 : 0.65
      const fraction = target === 0 && Math.abs(delta) < band ? 1 : -Math.expm1(-dt / tau)
      const step = Math.min(Math.abs(delta) * fraction, slew * dt)
      return current + Math.sign(delta) * step
    }
    const v = this.value, t = this.target
    this.value = {
      ev: follow(v.ev, t.ev, false), local: follow(v.local, t.local, false),
      red: follow(v.red, t.red, true), blue: follow(v.blue, t.blue, true)
    }
    if (fast && Math.abs(t.ev - this.value.ev) < 0.045 && Math.abs(t.local - this.value.local) < 0.045) this.acquiring = false
    return this.value
  }
}

// CPU reference for the two display shaders. Browser ingest produces
// full-range sRGB. Preserve black and apply the shoulder to the brightest
// channel so highly saturated colours also approach white level smoothly.
export function lightingCurve(linear: number[], gain: number, red: number, blue: number): number[] {
  const y = linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722
  const toe = clamp((y - 0.0004) / 0.0036, 0, 1)
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
