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
//
// readback (default on): the frame posted is a CPU copy of the rendered canvas
// (worker/cpu_frame.ts), not the canvas snapshot, so a software encoder does not
// have to read the GPU back inside its own time. The read-back is awaited here,
// before the next camera frame is taken: a slow read-back lowers the rate this
// sink delivers (counted in the renderer's fps and outputReadbackMs), it never
// queues frames. Off: the snapshot is posted, as before.

import type { Renderer } from '../renderer'
import { frameEntryTime } from '../latency'
import { toCpuFrame } from '../cpu_frame'

export interface MstgOutputOptions { readback?: boolean }

export function createMstgOutput(renderer: Renderer, port: MessagePort, onFrame: (entry?: number) => void, opts: MstgOutputOptions = {}): WritableStream<VideoFrame> {
  let readback = opts.readback ?? true
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
      if (readback) {
        try {
          const r = await toCpuFrame(out)
          out = r.frame
          if (r.copied) renderer.noteOutputReadback(r.ms)
          else { readback = false; console.warn('[longpipe] output frames cannot be copied to memory here; posting canvas snapshots') }
        } catch (err) {
          // The snapshot is still ours (toCpuFrame closes it only on success): post it,
          // and stop trying on this browser.
          readback = false
          console.warn('[longpipe] output read-back failed; posting canvas snapshots:', (err as Error)?.message ?? err)
        }
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
