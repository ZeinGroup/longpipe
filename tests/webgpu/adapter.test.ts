import { describe, it, expect, vi } from 'vitest'
import { describeAdapter, requestWebGPUAdapter, type AdapterSource } from '~/model/backends/webgpu/adapter.ts'

// Stand-in adapters: what each power preference gets, and whether it has shader-f16.
const adapter = (name: string, f16: boolean, info: Partial<GPUAdapterInfo> = {}) =>
  ({ name, features: new Set(f16 ? ['shader-f16'] : []), info } as unknown as GPUAdapter & { name: string })

function gpu(byPreference: Record<string, GPUAdapter | null | Error>): AdapterSource & { asked: (string | undefined)[] } {
  const asked: (string | undefined)[] = []
  return {
    asked,
    requestAdapter: vi.fn(async (options?: GPURequestAdapterOptions) => {
      asked.push(options?.powerPreference)
      const got = byPreference[options?.powerPreference ?? 'default']
      if (got instanceof Error) throw got
      return got ?? null
    }),
  }
}

describe('the WebGPU adapter Longpipe runs on', () => {
  it('is the default one when it has shader-f16, nothing else asked', async () => {
    const gpus = gpu({ default: adapter('integrated', true), 'high-performance': adapter('discrete', true) })
    expect(((await requestWebGPUAdapter(gpus)) as unknown as { name: string }).name).toBe('integrated')
    expect(gpus.asked).toEqual([undefined])
  })

  // A laptop with an NVIDIA Turing (default under Chrome on Linux, no shader-f16) and an Intel UHD 630 (shader-f16).
  it('is the low-power one when the default lacks shader-f16 and it has it', async () => {
    const gpus = gpu({ default: adapter('discrete', false), 'low-power': adapter('integrated', true), 'high-performance': adapter('discrete', false) })
    expect(((await requestWebGPUAdapter(gpus)) as unknown as { name: string }).name).toBe('integrated')
    expect(gpus.asked).toEqual([undefined, 'low-power'])
  })

  it('is the high-performance one when only it has shader-f16', async () => {
    const gpus = gpu({ default: adapter('integrated', false), 'low-power': adapter('integrated', false), 'high-performance': adapter('discrete', true) })
    expect(((await requestWebGPUAdapter(gpus)) as unknown as { name: string }).name).toBe('discrete')
    expect(gpus.asked).toEqual([undefined, 'low-power', 'high-performance'])
  })

  it('stays the default one when none has shader-f16, or another could not be asked', async () => {
    const none = gpu({ default: adapter('discrete', false), 'low-power': adapter('integrated', false), 'high-performance': adapter('discrete', false) })
    expect(((await requestWebGPUAdapter(none)) as unknown as { name: string }).name).toBe('discrete')
    const refusing = gpu({ default: adapter('discrete', false), 'low-power': new Error('no'), 'high-performance': null })
    expect(((await requestWebGPUAdapter(refusing)) as unknown as { name: string }).name).toBe('discrete')
  })

  it('is none without WebGPU, or without a default adapter', async () => {
    expect(await requestWebGPUAdapter(undefined)).toBeNull()
    expect(await requestWebGPUAdapter(gpu({ default: null }))).toBeNull()
    expect(await requestWebGPUAdapter(gpu({ default: new Error('lost') }))).toBeNull()
  })

  it("is told by its vendor and architecture, as its info gives them", () => {
    expect(describeAdapter(adapter('integrated', true, { vendor: 'intel', architecture: 'gen-9' }))).toBe('intel gen-9')
    expect(describeAdapter(adapter('apple', true, { vendor: 'apple', architecture: '' }))).toBe('apple')
    expect(describeAdapter(adapter('unknown', true))).toBe('')
    expect(describeAdapter(null)).toBe('')
  })
})
