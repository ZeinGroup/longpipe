// Guided upsampler, step 4, at output resolution: boxed (a, b) sampled bilinearly
// (align_corners = false) and applied to the full-resolution luminance,
// alpha = clamp(a·I + b, 0, 1), then optionally smoothstep(lo, hi) (when hi > lo).
// The matte's edges follow the camera's edges at full resolution instead of the
// model's grid, without reading or changing the temporal history.
// (f32 storage, f32 math)

struct Params {
    out_w : u32, out_h : u32, in_w : u32, in_h : u32,
    lo : f32, hi : f32, _p0 : u32, _p1 : u32,
}

@group(0) @binding(0) var<storage, read>       image_buf : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       coeffs_buf : array<vec4<f32>>;
@group(0) @binding(2) var<uniform>             params     : Params;
@group(0) @binding(3) var<storage, read_write> output_buf : array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= params.out_w || y >= params.out_h) { return; }
    let src_x = (f32(x) + 0.5) * f32(params.in_w) / f32(params.out_w) - 0.5;
    let src_y = (f32(y) + 0.5) * f32(params.in_h) / f32(params.out_h) - 0.5;
    let fx = src_x - floor(src_x); let fy = src_y - floor(src_y);
    let xa = u32(clamp(i32(floor(src_x)),     0, i32(params.in_w) - 1));
    let xb = u32(clamp(i32(floor(src_x)) + 1, 0, i32(params.in_w) - 1));
    let ya = u32(clamp(i32(floor(src_y)),     0, i32(params.in_h) - 1));
    let yb = u32(clamp(i32(floor(src_y)) + 1, 0, i32(params.in_h) - 1));
    let W = params.in_w;
    let ab = mix(mix(vec2<f32>(coeffs_buf[ya * W + xa].xy), vec2<f32>(coeffs_buf[ya * W + xb].xy), fx),
                 mix(vec2<f32>(coeffs_buf[yb * W + xa].xy), vec2<f32>(coeffs_buf[yb * W + xb].xy), fx), fy);
    let I = dot(vec3<f32>(image_buf[y * params.out_w + x].xyz), vec3<f32>(0.299, 0.587, 0.114));
    var a = clamp(ab.x * I + ab.y, 0.0, 1.0);
    if (params.hi > params.lo) {
        let t = clamp((a - params.lo) / (params.hi - params.lo), 0.0, 1.0);
        a = t * t * (3.0 - 2.0 * t);
    }
    output_buf[y * params.out_w + x] = vec4<f32>(f32(a), f32(0.0), f32(0.0), f32(0.0));
}
