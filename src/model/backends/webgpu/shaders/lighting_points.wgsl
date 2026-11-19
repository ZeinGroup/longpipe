// Each invocation refines one of the five keypoint peaks in the primary box.
@group(0) @binding(0) var<storage,read> hm: array<Pixel>;
@group(0) @binding(1) var<storage,read> boxes: array<Pixel>;
@group(0) @binding(2) var<uniform> size: vec2<u32>;
@group(0) @binding(3) var<storage,read_write> points: array<Pixel>;
fn score(p: vec2<i32>, k: u32) -> f32 {
    return f32(hm[(u32(p.y)*size.x+u32(p.x))*2+k/4][k%4]);
}
@compute @workgroup_size(5)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let box = vec4<f32>(boxes[0]);
    let side = box.z * f32(size.x);
    let lo = max(vec2<i32>(0), vec2<i32>(box.xy * vec2<f32>(size) - vec2<f32>(side)));
    let hi = min(vec2<i32>(size), vec2<i32>(box.xy * vec2<f32>(size) + vec2<f32>(side + 1.0)));
    var peak = 0.0;
    var pos = lo;
    for (var y=lo.y; y<hi.y; y++) { for (var x=lo.x; x<hi.x; x++) {
        let v = score(vec2<i32>(x,y),k);
        if (v>peak) { peak=v; pos=vec2<i32>(x,y); }
    }}
    var sum = vec2<f32>(0);
    var mass = 0.0;
    for (var y=-2; y<=2; y++) { for (var x=-2; x<=2; x++) {
        let p = clamp(pos+vec2<i32>(x,y),vec2<i32>(0),vec2<i32>(size) - vec2<i32>(1));
        let v = score(p,k);
        sum += v*(vec2<f32>(p) + vec2<f32>(0.5)); mass += v;
    }}
    points[k] = Pixel(vec4<f32>(sum/(max(mass,1e-6)*vec2<f32>(size)),min(box.w,peak),0));
}
