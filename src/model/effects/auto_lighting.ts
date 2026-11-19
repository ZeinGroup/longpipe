import type { Backend, DisplayLighting, LightAdjustment, Op, Tensor } from '../backend'
import { LightingController, METER_W, METER_H, METER_INTERVAL } from './lighting'

// Bound only to display ingestion. Model inputs, flow carriers, and uploaded
// background assets never see the correction or incur its sampling costs.
export class AutoLighting implements DisplayLighting {
  private readonly controller = new LightingController()
  private readonly raw: Tensor
  private readonly emptyGuide: Tensor
  private readonly emptyPoints: Tensor
  private readonly sceneMeter: Op
  private half?: Tensor
  private guide?: Op
  private subjectMeter?: Op
  private points?: Op
  private alpha: Tensor | null = null
  private useMatte = false
  private epoch = 0
  private pending = false
  private nextMeter = 0

  constructor(private readonly backend: Backend, private readonly width: number, private readonly height: number) {
    const [w, h] = width >= height ? [METER_W, METER_H] : [METER_H, METER_W]
    this.raw = backend.tensor(h, w, 4)
    this.emptyGuide = backend.tensor(h, w, 4)
    this.emptyPoints = backend.tensor(1, 5, 4)
    this.sceneMeter = backend.ops.LightingMeter(this.raw, this.emptyGuide)
  }

  setInputs(alpha: Tensor | null, hm: Tensor | null, box: Tensor | null): void {
    this.useMatte = alpha !== null
    if (!alpha) return
    if (alpha === this.alpha && hm === (this.points?.inputs[0] ?? null)
      && box === (this.points?.inputs[1] ?? null)) return
    this.alpha = alpha
    this.points = hm && box ? this.backend.ops.LightingPoints(hm, box) : undefined
    this.guide = this.backend.ops.LightingGuide(alpha, this.points?.output ?? this.emptyPoints, this.raw.w, this.raw.h)
    this.subjectMeter = this.backend.ops.LightingMeter(this.raw, this.guide.output)
  }

  reset(): void {
    this.epoch++
    this.controller.reset()
    this.nextMeter = 0
  }

  prepare(sample: (out: Tensor) => void): LightAdjustment | null {
    const now = performance.now()
    const tick = now >= this.nextMeter && !this.pending
    const levels = this.controller.step(now)
    const active = levels.ev !== 0 || levels.local !== 0 || levels.red !== 0 || levels.blue !== 0
    if (this.useMatte && (tick || levels.local > 0)) {
      if (tick) this.points?.run()
      this.guide!.run()
    }
    if (tick) {
      this.nextMeter = now + METER_INTERVAL
      sample(this.raw)
      const meter = this.useMatte ? this.subjectMeter! : this.sceneMeter
      meter.run()
      // Let the renderer submit its batch before recording the read. One
      // outstanding sample, and a late completion cannot undo a camera reset.
      this.pending = true
      const epoch = this.epoch, matte = this.useMatte
      void Promise.resolve().then(() => this.backend.readbackDeferred
        ? this.backend.readbackDeferred(meter.output) : this.backend.readback(meter.output))
        .then(tiles => { if (epoch === this.epoch) this.controller.observe(tiles, matte) })
        .catch(() => { /* context loss is handled by the worker */ })
        .finally(() => { this.pending = false })
    }
    if (!active) return null
    this.half ??= this.backend.tensor(Math.max(1, Math.floor(this.height / 2)), Math.max(1, Math.floor(this.width / 2)), 4)
    sample(this.half)
    return {
      guide: this.useMatte ? this.guide!.output : this.emptyGuide, half: this.half,
      gain: 2 ** levels.ev, localGain: this.useMatte ? 2 ** levels.local : 1, red: 2 ** levels.red, blue: 2 ** levels.blue
    }
  }
}
