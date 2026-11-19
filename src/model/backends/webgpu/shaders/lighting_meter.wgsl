@group(0) @binding(0) var<storage,read> raw: array<Pixel>;
@group(0) @binding(1) var<storage,read> guide: array<Pixel>;
@group(0) @binding(2) var<uniform> size: vec2<u32>;
@group(0) @binding(3) var<storage,read_write> tiles: array<Pixel>;
fn linearize(c: vec3<f32>) -> vec3<f32> {
    return select(c / vec3<f32>(12.92),pow((c + vec3<f32>(0.055)) / vec3<f32>(1.055),vec3<f32>(2.4)),c>vec3<f32>(0.04045));
}
@compute @workgroup_size(8,4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= size / vec2<u32>(6))) { return; }
    var sums = vec3<f32>(0); var masses = vec3<f32>(0);
    var gains = vec2<f32>(0);
    var total = 0.0; var neutral = 0.0;
    var chromaticity = vec2<f32>(0);
    for (var y=0u; y<6; y++) { for (var x=0u; x<6; x++) {
        let i = (gid.y*6+y)*size.x+gid.x*6+x;
        let encoded = vec3<f32>(raw[i].rgb);
        let rgb = linearize(encoded);
        let lum = dot(rgb,vec3<f32>(0.2126,0.7152,0.0722));
        let logY = log2(max(lum,0.001));
        total += logY;
        chromaticity += rgb.rb / vec2<f32>(max(0.015, rgb.r + rgb.g + rgb.b));
        if (lum < 0.004 || max(max(encoded.r,encoded.g),encoded.b) >= 0.975) { continue; }
        let g = vec4<f32>(guide[i]);
        let weights = vec3<f32>(1,g.yz);
        sums += weights*logY; masses += weights;
        let ratios = log2(vec2<f32>(max(rgb.g, 0.001)) / max(rgb.rb, vec2<f32>(0.001)));
        let chroma = max(abs(ratios.x),abs(ratios.y));
        let w = (1.0-smoothstep(0.23,0.65,chroma))*smoothstep(0.015,0.055,lum)*(1.0-g.y);
        gains += ratios*w; neutral += w;
    }}
    let o = (gid.y*(size.x/6)+gid.x)*3;
    tiles[o] = Pixel(vec4<f32>(sums,total/36.0));
    tiles[o+1] = Pixel(vec4<f32>(masses,neutral));
    tiles[o+2] = Pixel(vec4<f32>(gains,chromaticity/36.0));
}
