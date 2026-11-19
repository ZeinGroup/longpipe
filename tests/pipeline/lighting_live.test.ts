import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { Pipeline } from '~/pipeline/index'
import { TUNING } from '~/pipeline/tuning'
import { SMALL, stubWeights, restoreFetch, waitUntil } from '../helpers/pipeline'

function greyCamera(level: number) {
  const canvas = document.createElement('canvas')
  canvas.width = 160; canvas.height = 90
  const ctx = canvas.getContext('2d')!
  let tick = 0
  const draw = () => {
    ctx.fillStyle = `rgb(${level},${level},${level})`
    ctx.fillRect(0, 0, 160, 90)
    ctx.fillStyle = tick++ % 2 ? 'black' : 'white'
    ctx.fillRect(0, 0, 2, 2)
  }
  draw()
  const stream = canvas.captureStream(24)
  const timer = setInterval(draw, 1000 / 24)
  return { stream, stop() { clearInterval(timer); stream.getTracks().forEach(t => t.stop()) } }
}

beforeEach(() => stubWeights())
afterEach(restoreFetch)

describe.each(['webgl', 'webgpu'] as const)('live lighting on %s', backend => {
  it('lifts a dim outgoing track with no background, toggles in place and resets on a camera switch', async ctx => {
    TUNING.forceBackend = backend
    const dim = greyCamera(50), bright = greyCamera(165)
    const p = new Pipeline(dim.stream, { preset: SMALL, background: 'none', lighting: true })
    const video = document.createElement('video')
    const probe = new OffscreenCanvas(1, 1).getContext('2d', { willReadFrequently: true })!
    const level = () => {
      if (!video.videoWidth) return -1
      probe.drawImage(video, 80, 45, 1, 1, 0, 0, 1, 1)
      return probe.getImageData(0, 0, 1, 1).data[0]
    }
    try {
      const info = await p.ready
      if (info.backend !== (backend === 'webgl' ? 'webgl2' : 'webgpu')) return ctx.skip()
      const track = p.stream.getVideoTracks()[0]
      video.muted = true
      video.srcObject = p.stream
      await video.play()
      await waitUntil(() => level() > 64, 10000, 'lit output')
      await p.setLighting(false)
      await waitUntil(() => Math.abs(level() - 50) < 4, 6000, 'raw output')
      await p.setBackground('blur')
      await p.setLighting(true)
      await waitUntil(() => level() > 64, 10000, 'lit blur output')
      await p.setInputStream(bright.stream)
      await waitUntil(() => Math.abs(level() - 165) < 4, 6000, 'well-lit output')
      expect(p.stream.getVideoTracks()[0]).toBe(track)
    } finally {
      video.pause()
      video.srcObject = null
      p.destroy()
      dim.stop()
      bright.stop()
    }
  }, 60000)
})
