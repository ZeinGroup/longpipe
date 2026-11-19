// Guided upsampler, step 3: per-pixel linear model of the matte in the guide's luminance,
// a = cov(I, p) / (var(I) + eps), b = mean(p) − a·mean(I), from boxed (I, p, I·p, I·I).
// (f32 storage, f32 math)

struct Params {
    n : u32, eps : f32, _p1 : u32, _p2 : u32,
}

@group(0) @binding(0) var<storage, read>       input_buf : array<vec4<f32>>;
@group(0) @binding(1) var<uniform>             params     : Params;
@group(0) @binding(2) var<storage, read_write> output_buf : array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    let m = vec4<f32>(input_buf[idx]);   // (mean I, mean p, mean I·p, mean I·I)
    let a = (m.z - m.x * m.y) / (max(m.w - m.x * m.x, 0.0) + params.eps);
    output_buf[idx] = vec4<f32>(vec4<f32>(a, m.y - a * m.x, 0.0, 0.0));
}
