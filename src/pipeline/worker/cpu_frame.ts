// A rendered frame in CPU memory for the output track.
//
// A VideoFrame made from the canvas is a snapshot of a GPU surface. A software
// video encoder needs the pixels in memory, so the browser has to wait for the
// GPU, read the frame back and convert it inside the encoder's own time, which
// its CPU-overuse detection then counts against the resolution it sends. Here
// the worker reads the frame back itself (VideoFrame.copyTo, which waits for the
// GPU asynchronously) and hands the track a frame over those bytes: same size,
// timestamp and colour, alpha marked unused, the buffer transferred into the new
// frame (no second copy). The wait falls on the filter: while a read-back is in
// progress the sink takes no new camera frame.

const OPAQUE: Partial<Record<VideoPixelFormat, VideoPixelFormat>> = { RGBA: 'RGBX', BGRA: 'BGRX' }

export interface CpuFrameResult {
  frame:   VideoFrame
  copied:  boolean   // false: the snapshot itself (the browser could not copy it out)
  ms:      number    // read-back time, snapshot to CPU frame
}

export async function toCpuFrame(snapshot: VideoFrame): Promise<CpuFrameResult> {
  const t0 = performance.now()
  const format = snapshot.format
  if (!format) return { frame: snapshot, copied: false, ms: 0 }
  const rect = snapshot.visibleRect
  const width = rect?.width ?? snapshot.codedWidth
  const height = rect?.height ?? snapshot.codedHeight
  const buffer = new ArrayBuffer(snapshot.allocationSize())
  const layout = await snapshot.copyTo(buffer)
  const init: VideoFrameBufferInit & { transfer?: ArrayBuffer[] } = {
    format:        OPAQUE[format] ?? format,
    codedWidth:    width,
    codedHeight:   height,
    timestamp:     snapshot.timestamp,
    displayWidth:  snapshot.displayWidth,
    displayHeight: snapshot.displayHeight,
    colorSpace:    snapshot.colorSpace.toJSON(),
    layout,
    transfer:      [buffer],
  }
  if (snapshot.duration !== null) init.duration = snapshot.duration
  const frame = new VideoFrame(buffer, init)
  snapshot.close()
  return { frame, copied: true, ms: performance.now() - t0 }
}
