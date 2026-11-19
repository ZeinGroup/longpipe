enable f16;

// Normalized convolution, part 2: rgb / max(a, eps) after blurring (rgb·w, w). Where the
// blur footprint is all subject (a -> 0) it fades to black, which only shows
// where the subject's own alpha is ~1 anyway.
// Elementwise over n pixels of 4-channel tensors (f16 storage).

struct Params {
    n     : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
}

@group(0) @binding(0) var<storage, read>       input_buf : array<vec4<f16>>;
@group(0) @binding(1) var<uniform>             params     : Params;
@group(0) @binding(2) var<storage, read_write> output_buf : array<vec4<f16>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    let t = vec4<f32>(input_buf[idx]);
    output_buf[idx] = vec4<f16>(vec3<f16>(t.xyz / max(t.w, 2e-3)), f16(1.0));
}
