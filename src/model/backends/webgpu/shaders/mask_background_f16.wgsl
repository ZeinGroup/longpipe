enable f16;

// Background-only blur input (normalized convolution, part 1): the image weighted by the
// background's share w = 1 - alpha, with w itself in .a. Run through the blur
// pyramid and divided back (Unpremultiply), the blurred background takes no
// colour from the subject: no halo of the person around the silhouette.
// Elementwise over n pixels of 4-channel tensors (f16 storage).

struct Params {
    n     : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
}

@group(0) @binding(0) var<storage, read>       image_buf : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       alpha_buf : array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             params     : Params;
@group(0) @binding(3) var<storage, read_write> output_buf : array<vec4<f16>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }
    let w = 1.0 - clamp(f32(alpha_buf[idx].x), 0.0, 1.0);
    let c = vec3<f32>(image_buf[idx].xyz);
    output_buf[idx] = vec4<f16>(vec3<f16>(c * w), f16(w));
}
