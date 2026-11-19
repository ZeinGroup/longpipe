import { describe, expect, it } from 'vitest'
import { toCpuFrame } from '~/pipeline/worker/cpu_frame'

// Real frames: a canvas snapshot from a WebGL2 and a WebGPU OffscreenCanvas comes back
// as a frame in memory with the same pixels, size and timestamp, alpha marked unused.
async function drawn(kind: 'webgl2' | 'webgpu'): Promise<OffscreenCanvas | null> {
  const canvas = new OffscreenCanvas(64, 36)
  if (kind === 'webgl2') {
    const gl = canvas.getContext('webgl2')
    if (!gl) return null
    gl.enable(gl.SCISSOR_TEST)
    gl.scissor(0, 0, 32, 36); gl.clearColor(1, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT)
    gl.scissor(32, 0, 32, 36); gl.clearColor(0, 0, 1, 1); gl.clear(gl.COLOR_BUFFER_BIT)
    return canvas
  }
  const adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice()
  const ctx = canvas.getContext('webgpu')
  if (!device || !ctx) return null
  ctx.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'premultiplied' })
  const enc = device.createCommandEncoder()
  const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 1, b: 0, a: 1 } }] })
  pass.end()
  device.queue.submit([enc.finish()])
  await device.queue.onSubmittedWorkDone()
  return canvas
}

describe.each(['webgl2', 'webgpu'] as const)('CPU output frames from a %s canvas', kind => {
  it('keep the pixels, size and timestamp of the snapshot', async ctx => {
    const canvas = await drawn(kind)
    if (!canvas) return ctx.skip()
    const snap = new VideoFrame(canvas, { timestamp: 4242 })
    const ref = new Uint8Array(snap.allocationSize())
    await snap.copyTo(ref)
    const format = snap.format
    const { frame, copied } = await toCpuFrame(snap)
    try {
      expect(copied).toBe(true)
      expect(frame.timestamp).toBe(4242)
      expect([frame.codedWidth, frame.codedHeight]).toEqual([64, 36])
      expect(frame.format).toBe(format === 'RGBA' ? 'RGBX' : format === 'BGRA' ? 'BGRX' : format)
      const got = new Uint8Array(frame.allocationSize())
      await frame.copyTo(got)
      // compare colour channels only (the copy's alpha is declared unused)
      for (let i = 0; i < got.length; i += 4) for (let c = 0; c < 3; c++) expect(got[i + c]).toBe(ref[i + c])
    } finally { frame.close() }
  })
})
