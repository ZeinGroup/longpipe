// MSTG output adapter.
//
// Worker renders to its own OffscreenCanvas, wraps each rendered frame in a
// new VideoFrame, and posts it (transferred, zero-copy) to the output relay
// worker, which owns the MediaStreamTrackGenerator's writable (see
// transports/relay_worker.ts for why the writable no longer lives here).
//
// Pipe: inputReadable → this sink. The relay drops a frame if the generator
// hasn't accepted the previous one yet, so nothing queues after the GPU; the
// sink itself keeps at most one frame in flight on the GPU (waitForGpu).

import type { Renderer } from '../renderer'
import { frameEntryTime } from '../latency'

export function createMstgOutput(renderer: Renderer, port: MessagePort, onFrame: (entry?: number) => void): WritableStream<VideoFrame> {
  return new WritableStream<VideoFrame>({
    async write(frame) {
      const entry = frameEntryTime(frame, performance.now())
      let out: VideoFrame
      try {
        const src = await renderer.prepareSource(frame)
        try { renderer.process(frame, src) } finally { if (src !== frame) src.close() }
        out = new VideoFrame(renderer.canvas, { timestamp: frame.timestamp })
        renderer.markSubmitted()
      } finally {
        // Closed even when processing throws (GPU lost mid-frame): an unclosed
        // camera frame pins a capture buffer until GC.
        frame.close()
      }
      port.postMessage({ frame: out }, [out])
      onFrame(entry)
      // As on bitmap-shuttle and transfer-capture: don't take the next camera
      // frame until the GPU has finished this one. The relay dropping frames it
      // can't write bounds the queue AFTER the GPU, not the GPU's own queue: on a
      // GPU slower than the camera, work piled up there (output fell to <1 fps
      // with seconds of delay while the renderer, counting submissions, kept
      // reporting 30 fps and the adaptive controller never stepped down).
      await renderer.waitForGpu(1)
    },
    close() {
      port.close()
    },
  })
}
