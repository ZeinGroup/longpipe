// Fused into display ingestion (either a texture_external or the 2D fallback).
struct LightParams { gain: vec4<f32>, dims: vec4<u32>, wideDims: vec4<u32> } // guide wh, half wh; wide wh
@group(0) @binding(4) var<uniform> light: LightParams;
@group(0) @binding(5) var<storage,read> guide: array<Pixel>;
@group(0) @binding(6) var<storage,read> halfFrame: array<Pixel>;
@group(0) @binding(7) var<storage,read> wideFrame: array<Pixel>;
// Colour from the 1/16-resolution average: strength per stop of lift above unity, its cap,
// and the luma-flatness and colour-agreement gates (encoded units).
const WIDE_SLOPE: f32 = 0.4; const WIDE_MAX: f32 = 0.95; const WIDE_FLAT_LO: f32 = 0.05; const WIDE_FLAT_HI: f32 = 0.12;
const WIDE_AGREE_LO: f32 = 0.03; const WIDE_AGREE_HI: f32 = 0.08;
fn localWeight(uv: vec2<f32>) -> f32 {
    if (light.gain.y == 1.0) { return 0.0; }
    let size = light.dims.xy;
    let pos = clamp(uv * vec2<f32>(size) - vec2<f32>(0.5),vec2<f32>(0),vec2<f32>(size - vec2<u32>(1)));
    let a = vec2<u32>(pos); let b = min(a + vec2<u32>(1), size - vec2<u32>(1));
    let f = fract(pos);
    return mix(mix(f32(guide[a.y*size.x+a.x].x),f32(guide[a.y*size.x+b.x].x),f.x),
                     mix(f32(guide[b.y*size.x+a.x].x),f32(guide[b.y*size.x+b.x].x),f.x),f.y);
}
// The half- and 1/16-resolution references, bilinear (smooth local averages, no blocks).
fn halfSample(uv: vec2<f32>) -> vec3<f32> {
    let size = light.dims.zw;
    let pos = clamp(uv * vec2<f32>(size) - vec2<f32>(0.5),vec2<f32>(0),vec2<f32>(size - vec2<u32>(1)));
    let a = vec2<u32>(pos); let b = min(a + vec2<u32>(1), size - vec2<u32>(1));
    let f = fract(pos);
    return mix(mix(vec3<f32>(halfFrame[a.y*size.x+a.x].rgb),vec3<f32>(halfFrame[a.y*size.x+b.x].rgb),f.x),
               mix(vec3<f32>(halfFrame[b.y*size.x+a.x].rgb),vec3<f32>(halfFrame[b.y*size.x+b.x].rgb),f.x),f.y);
}
fn wideSample(uv: vec2<f32>) -> vec3<f32> {
    let size = light.wideDims.xy;
    let pos = clamp(uv * vec2<f32>(size) - vec2<f32>(0.5),vec2<f32>(0),vec2<f32>(size - vec2<u32>(1)));
    let a = vec2<u32>(pos); let b = min(a + vec2<u32>(1), size - vec2<u32>(1));
    let f = fract(pos);
    return mix(mix(vec3<f32>(wideFrame[a.y*size.x+a.x].rgb),vec3<f32>(wideFrame[a.y*size.x+b.x].rgb),f.x),
               mix(vec3<f32>(wideFrame[b.y*size.x+a.x].rgb),vec3<f32>(wideFrame[b.y*size.x+b.x].rgb),f.x),f.y);
}
fn correctLighting(src: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let weight = localWeight(uv);
    let gain = light.gain.x*mix(1.0,light.gain.y,weight);
    let halfRgb = halfSample(uv);
    let luma = vec3<f32>(0.2126,0.7152,0.0722);
    let delta = halfRgb-src;
    let dy = dot(delta,luma);
    let y = dot(src,luma);
    var denoise = min(0.6,(light.gain.x-1.0)*0.45)*(1.0-smoothstep(0.025,0.09,abs(dy)))
                * (1.0-smoothstep(0.18,0.48,y));
    let chroma = delta-vec3<f32>(dy);
    denoise *= 1.0-smoothstep(0.065,0.16,max(abs(chroma.r),max(abs(chroma.g),abs(chroma.b))));
    // Shadow luma: the lift multiplies sensor noise by the curve's slope, steepest in the
    // darks. Flat dark pixels move toward the local average; a step well above the noise
    // (an edge) and everything above the shadows keep full resolution.
    let smoothY = clamp((gain-1.0)*0.3,0.0,0.7)*(1.0-smoothstep(0.02,0.06,abs(dy)))
                * (1.0-smoothstep(0.12,0.40,y));
    // Webcam colour noise comes in blotches (4:2:0 chroma and its compression blocks, 16 px
    // at 720p) that a half-resolution average cannot reach. Where the lift is strong, take the
    // colour of the 1/16-resolution average (a few blocks across) instead, but only where the area is flat in luma
    // at that scale (no outline, no edge) and its colour agrees with the fine average's
    // (real colour structure stays). Luma is untouched; the mean colour, so saturation, stays.
    let wideRgb = wideSample(uv);
    let dW = wideRgb-src;
    let dyW = dot(dW,luma);
    let dc = (halfRgb-vec3<f32>(dot(halfRgb,luma))) - (wideRgb-vec3<f32>(dyW+dot(src,luma)));
    let wideC = clamp((gain-1.0)*WIDE_SLOPE,0.0,WIDE_MAX)*(1.0-smoothstep(WIDE_FLAT_LO,WIDE_FLAT_HI,abs(dyW)))
              * (1.0-smoothstep(WIDE_AGREE_LO,WIDE_AGREE_HI,max(abs(dc.r),max(abs(dc.g),abs(dc.b)))));
    var c = clamp(src+mix(denoise*chroma,dW-vec3<f32>(dyW),wideC)+vec3<f32>(smoothY*dy),vec3<f32>(0),vec3<f32>(1));
    let linear = select(c / vec3<f32>(12.92),pow((c + vec3<f32>(0.055)) / vec3<f32>(1.055),vec3<f32>(2.4)),c>vec3<f32>(0.04045));
    let lift = mix(1.0,gain,smoothstep(0.0004,0.004,dot(linear,luma)));
    let rgb = linear*vec3<f32>(light.gain.z,1,light.gain.w);
    let peak = max(max(rgb.r,rgb.g),rgb.b);
    var scale = lift/(1.0+(lift-1.0)*peak);
    let p = peak*scale;
    let limit = max(1.0,max(light.gain.z,light.gain.w));
    let curvature = (limit-1.0)/(0.2*(limit-0.8));
    let excess = max(0.0,p-0.8);
    if (p>0.8) { scale *= (0.8+excess/(1.0+curvature*excess))/p; }
    c = rgb*scale;
    return select(12.92*c,1.055 * pow(c, vec3<f32>(1.0/2.4)) - vec3<f32>(0.055),c>vec3<f32>(0.0031308));
}
