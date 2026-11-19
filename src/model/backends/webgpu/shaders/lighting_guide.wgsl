@group(0) @binding(0) var<storage,read> matte: array<Pixel>;
@group(0) @binding(1) var<storage,read> points: array<Pixel>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>; // matte wh, guide wh
@group(0) @binding(3) var<storage,read_write> guide: array<Pixel>;
fn alpha(uv: vec2<f32>) -> f32 {
    let p = vec2<u32>(clamp(vec2<i32>(uv*vec2<f32>(dims.xy)),vec2<i32>(0),vec2<i32>(dims.xy) - vec2<i32>(1)));
    return f32(matte[p.y*dims.x+p.x].x);
}
fn disk(p: vec2<f32>, c: vec2<f32>, radius: f32) -> f32 {
    return 1.0-smoothstep(0.45*radius,radius,length(p-c));
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= dims.zw)) { return; }
    let grid = vec2<f32>(dims.zw);
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5))/grid;
    var inner = 1.0;
    var outer = 0.0;
    for (var y=-1; y<=1; y++) { for (var x=-1; x<=1; x++) {
        let d = vec2<f32>(f32(x),f32(y))/grid;
        inner = min(inner,alpha(uv+1.5*d));
        outer += alpha(uv+3.0*d);
    }}
    let person = smoothstep(0.75,0.98,inner)*smoothstep(0.65,1.0,outer/9.0);
    let le = vec4<f32>(points[0]); let re = vec4<f32>(points[1]);
    let nose = vec4<f32>(points[2]);
    let lm = vec4<f32>(points[3]); let rm = vec4<f32>(points[4]);
    let eyes = (re.xy-le.xy)*grid;
    let span = max(length(eyes),0.01);
    let axis = eyes / vec2<f32>(span);
    var down = vec2<f32>(-axis.y,axis.x);
    let origin = (le.xy+re.xy)*0.5;
    let mouth = ((lm.xy+rm.xy)*0.5-origin)* grid / vec2<f32>(span);
    down *= sign(dot(mouth,down));
    let delta = (uv-origin)* grid / vec2<f32>(span);
    let p = vec2<f32>(dot(delta,axis),dot(delta,down));
    let height = dot(mouth,down);
    var conf = min(min(le.z,re.z),min(nose.z,min(lm.z,rm.z)));
    conf = smoothstep(0.15,0.30,conf)*smoothstep(1.3,3.0,span)
         * smoothstep(0.45,0.8,height)*(1.0-smoothstep(1.6,2.1,height));
    let cheeks = disk(vec2<f32>(abs(p.x),p.y),vec2<f32>(0.49,0.58*height),0.34);
    let forehead = disk(p,vec2<f32>(0,-0.46),0.40);
    let excluded = max(disk(vec2<f32>(abs(p.x),p.y),vec2<f32>(0.5,0),0.31),
                       disk((uv-(lm.xy+rm.xy)*0.5)* grid / vec2<f32>(span),vec2<f32>(0),0.5));
    let face = conf*max(cheeks,forehead)*(1.0-excluded);
    let broad = conf*(1.0-smoothstep(0.42,1.22,length(p-vec2<f32>(0,0.28))));
    guide[gid.y*dims.z+gid.x] = Pixel(vec4<f32>(person*(0.65+0.35*broad),person,person*face,0));
}
