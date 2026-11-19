import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { toCpuFrame } from '~/pipeline/worker/cpu_frame'
import { createMstgOutput } from '~/pipeline/worker/adapters/output_mstg'

// A stand-in VideoFrame: canvas snapshots (format RGBA, GPU-like) and frames over a
// buffer, enough to check what the generator sink posts.
let copyGate: Promise<void> | null = null
let copyFails = false
class FakeFrame {
  closed = false
  format: VideoPixelFormat | null
  codedWidth: number; codedHeight: number; displayWidth: number; displayHeight: number
  visibleRect: { width: number; height: number }
  timestamp: number; duration: number | null = null
  colorSpace = { toJSON: () => ({ primaries: 'bt709', transfer: 'iec61966-2-1', matrix: 'rgb', fullRange: true }) }
  init: Record<string, unknown>
  data: ArrayBuffer | null
  constructor(source: unknown, init: Record<string, unknown>) {
    this.init = init
    const fromBuffer = source instanceof ArrayBuffer
    this.data = fromBuffer ? source as ArrayBuffer : null
    const c = source as { width?: number; height?: number }
    this.format = fromBuffer || 'format' in init ? init.format as VideoPixelFormat | null : 'RGBA'
    this.codedWidth = fromBuffer ? init.codedWidth as number : c.width ?? 4
    this.codedHeight = fromBuffer ? init.codedHeight as number : c.height ?? 2
    this.displayWidth = this.codedWidth; this.displayHeight = this.codedHeight
    this.visibleRect = { width: this.codedWidth, height: this.codedHeight }
    this.timestamp = init.timestamp as number
  }
  allocationSize() { return this.codedWidth * this.codedHeight * 4 }
  async copyTo(dest: ArrayBuffer) {
    if (copyGate) await copyGate
    if (copyFails) throw new Error('copy failed')
    new Uint8Array(dest).fill(7)
    return [{ offset: 0, stride: this.codedWidth * 4 }]
  }
  close() { this.closed = true }
}

beforeEach(() => { vi.stubGlobal('VideoFrame', FakeFrame); copyGate = null; copyFails = false })
afterEach(() => { vi.unstubAllGlobals() })

describe('CPU frames for the output track', () => {
  it('copies a snapshot to memory with the same size, timestamp and colour, alpha unused', async () => {
    const snap = new FakeFrame({ width: 6, height: 4 }, { timestamp: 1234 }) as unknown as VideoFrame
    const { frame, copied } = await toCpuFrame(snap)
    const f = frame as unknown as FakeFrame
    expect(copied).toBe(true)
    expect((snap as unknown as FakeFrame).closed).toBe(true)
    expect(f.init).toMatchObject({ format: 'RGBX', codedWidth: 6, codedHeight: 4, timestamp: 1234, colorSpace: { matrix: 'rgb', fullRange: true } })
    expect((f.init.transfer as ArrayBuffer[])[0]).toBe(f.data)   // the buffer is moved, not copied again
    expect(new Uint8Array(f.data!)[0]).toBe(7)
  })
  it('hands back the snapshot untouched when it has no readable format', async () => {
    const snap = new FakeFrame({ width: 6, height: 4 }, { timestamp: 1, format: null }) as unknown as VideoFrame
    const r = await toCpuFrame(snap)
    expect(r.copied).toBe(false)
    expect(r.frame).toBe(snap)
    expect((snap as unknown as FakeFrame).closed).toBe(false)
  })
})

describe('the generator sink', () => {
  function rig(opts?: { readback?: boolean }) {
    const posted: FakeFrame[] = []
    const renderer = {
      canvas: { width: 8, height: 4 },
      prepareSource: async (f: VideoFrame) => f,
      process: vi.fn(),
      markSubmitted: vi.fn(),
      waitForGpu: vi.fn(async () => {}),
      noteOutputReadback: vi.fn(),
    }
    const port = { postMessage: vi.fn((m: { frame: FakeFrame }) => { posted.push(m.frame) }), close: vi.fn() }
    const sink = createMstgOutput(renderer as never, port as unknown as MessagePort, () => {}, opts)
    return { renderer, posted, writer: sink.getWriter() }
  }
  const camera = (t: number) => new FakeFrame({ width: 8, height: 4 }, { timestamp: t }) as unknown as VideoFrame

  it('posts a CPU frame by default, counts its read-back, and closes the camera frame', async () => {
    const { renderer, posted, writer } = rig()
    const cam = camera(40)
    await writer.write(cam)
    expect((cam as unknown as FakeFrame).closed).toBe(true)
    expect(posted).toHaveLength(1)
    expect(posted[0].data).not.toBeNull()
    expect(posted[0].timestamp).toBe(40)
    expect(renderer.noteOutputReadback).toHaveBeenCalledTimes(1)
  })
  it('takes no new camera frame until the read-back of the last one is done', async () => {
    const { posted, writer } = rig()
    let open!: () => void
    copyGate = new Promise(r => { open = r })
    let done = false
    const w = writer.write(camera(1)).then(() => { done = true })
    await new Promise(r => setTimeout(r, 0))
    expect(done).toBe(false)
    expect(posted).toHaveLength(0)
    open(); await w
    expect(posted).toHaveLength(1)
  })
  it('posts the snapshots when turned off, and falls back to them if the copy fails', async () => {
    const off = rig({ readback: false })
    await off.writer.write(camera(1))
    expect(off.posted[0].data).toBeNull()
    expect(off.renderer.noteOutputReadback).not.toHaveBeenCalled()
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const failing = rig()
    copyFails = true
    await failing.writer.write(camera(2))
    copyFails = false
    await failing.writer.write(camera(3))
    expect(failing.posted.map(f => f.data)).toEqual([null, null])   // stays on snapshots after a failure
    expect(failing.posted.every(f => !f.closed)).toBe(true)
    expect(warn).toHaveBeenCalledTimes(1)
    warn.mockRestore()
  })
})
