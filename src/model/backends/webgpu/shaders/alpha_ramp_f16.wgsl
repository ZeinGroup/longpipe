enable f16;

// Display-alpha ramp: a = smoothstep(lo, hi, alpha.x). Tightens the soft edge the
// compositor blends over, so less of the original (sharp) room shows through the
// transition around a subject. Applied to the display alpha only (after the
// temporal stabilizer, never fed back into it). alpha in .x; output (a, 0, 0, 0).
// Elementwise over n pixels of 4-channel tensors (f16 storage).

struct Params {
    n    : u32,
    lo   : f32,
    hi   : f32,
    _pad : u32,
}

@group(0) @binding(0) var<storage, read>       input_buf : array<vec4<f16>>;
@group(0) @binding(1) var<uniform>             params     : Params;
@group(0) @binding(2) var<storage, read_write> output_buf : array<vec4<f16>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    let a = f32(input_buf[idx].x);
    let t = clamp((a - params.lo) / max(params.hi - params.lo, 1e-4), 0.0, 1.0);
    output_buf[idx] = vec4<f16>(f16(t * t * (3.0 - 2.0 * t)), f16(0.0), f16(0.0), f16(0.0));
}
