// Guided upsampler, step 1 (fast guided filter, He et al. 2015 / Wu et al. 2018): at the
// model's grid, pack (I, p, I·p, I·I) — I the guide's luminance (the camera frame
// resampled to the model's input), p the matte (.x).
// (f32 storage, f32 math)

struct Params {
    n : u32, _p0 : u32, _p1 : u32, _p2 : u32,
}

@group(0) @binding(0) var<storage, read>       guide_buf : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       alpha_buf : array<vec4<f32>>;
@group(0) @binding(2) var<uniform>             params     : Params;
@group(0) @binding(3) var<storage, read_write> output_buf : array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    let I = dot(vec3<f32>(guide_buf[idx].xyz), vec3<f32>(0.299, 0.587, 0.114));
    let p = f32(alpha_buf[idx].x);
    output_buf[idx] = vec4<f32>(vec4<f32>(I, p, I * p, I * I));
}
