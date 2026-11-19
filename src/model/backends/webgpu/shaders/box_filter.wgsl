// Guided upsampler, step 2: separable box mean (2·radius + 1 taps, edge replicated)
// along one axis, all four channels.
// (f32 storage, f32 math)

struct Params {
    w : u32, h : u32, radius : u32, vertical : u32,
}

@group(0) @binding(0) var<storage, read>       input_buf : array<vec4<f32>>;
@group(0) @binding(1) var<uniform>             params     : Params;
@group(0) @binding(2) var<storage, read_write> output_buf : array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= params.w || y >= params.h) { return; }
    var s = vec4<f32>(0.0);
    let r = i32(params.radius);
    for (var k = -r; k <= r; k++) {
        var qx = i32(x); var qy = i32(y);
        if (params.vertical == 1u) { qy = clamp(qy + k, 0, i32(params.h) - 1); } else { qx = clamp(qx + k, 0, i32(params.w) - 1); }
        s += vec4<f32>(input_buf[u32(qy) * params.w + u32(qx)]);
    }
    output_buf[y * params.w + x] = vec4<f32>(s / f32(2 * r + 1));
}
