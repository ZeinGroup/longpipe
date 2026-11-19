// Which WebGPU adapter Longpipe runs on.
//
// The default adapter, unless it lacks `shader-f16` (the half-precision shaders
// and the f16 weight packs) and another adapter on the same machine has it. On
// a laptop with two GPUs the default can be the discrete one without it (e.g.
// an NVIDIA Turing under Chrome on Linux) while the integrated one has it
// (an Intel UHD 630 there). Then it asks again by power preference, low-power
// first, then high-performance, and takes the first that has `shader-f16`;
// failing that, the default. Where the default already has it, nothing else is
// asked.
//
// Exported so a page can ask the same question and know where the worker will
// land (both run this same function against the same GPUs).

export type AdapterSource = { requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null> }

const PREFERENCES: readonly GPUPowerPreference[] = ['low-power', 'high-performance']

export async function requestWebGPUAdapter(
  gpu: AdapterSource | undefined = (globalThis.navigator as Navigator | undefined)?.gpu,
): Promise<GPUAdapter | null> {
  if (!gpu) return null
  const ask = (options?: GPURequestAdapterOptions) => gpu.requestAdapter(options).catch(() => null)
  const first = await ask()
  if (!first || first.features.has('shader-f16')) return first
  for (const powerPreference of PREFERENCES) {
    const other = await ask({ powerPreference })
    if (other?.features.has('shader-f16')) return other
  }
  return first
}

// "vendor architecture" as the adapter's info gives them ('' where it says nothing).
export function describeAdapter(adapter: GPUAdapter | null | undefined): string {
  const info = (adapter as { info?: Partial<GPUAdapterInfo> } | null | undefined)?.info
  return [info?.vendor, info?.architecture].filter(Boolean).join(' ')
}
