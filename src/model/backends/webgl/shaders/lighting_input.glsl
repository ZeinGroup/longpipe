// Fused into the display Input pass. Both tensors have top-down orientation.
uniform highp sampler2D u_guide, u_half, u_wide;
// Colour from the 1/16-resolution average: strength per stop of lift above unity, its cap,
// and the luma-flatness and colour-agreement gates (encoded units).
const float WIDE_SLOPE = 0.4, WIDE_MAX = 0.95, WIDE_FLAT_LO = 0.05, WIDE_FLAT_HI = 0.12, WIDE_AGREE_LO = 0.03, WIDE_AGREE_HI = 0.08;
uniform vec4 u_gain; // global, local, red, blue (linear gains)
float localWeight(vec2 uv) {
    if (u_gain.y == 1.0) return 0.0;
    ivec2 size = textureSize(u_guide,0);
    vec2 pos = clamp(uv*vec2(size)-0.5,vec2(0),vec2(size-1));
    ivec2 a = ivec2(pos), b = min(a+1,size-1);
    vec2 f = fract(pos);
    return mix(mix(texelFetch(u_guide,a,0).x,texelFetch(u_guide,ivec2(b.x,a.y),0).x,f.x),
                       mix(texelFetch(u_guide,ivec2(a.x,b.y),0).x,texelFetch(u_guide,b,0).x,f.x),f.y);
}
// The half- and 1/16-resolution references, bilinear (smooth local averages, no blocks).
vec3 bilinear(highp sampler2D t, vec2 uv) {
    ivec2 size = textureSize(t,0);
    vec2 pos = clamp(uv*vec2(size)-0.5,vec2(0),vec2(size-1));
    ivec2 a = ivec2(pos), b = min(a+1,size-1);
    vec2 f = fract(pos);
    return mix(mix(texelFetch(t,a,0).rgb,texelFetch(t,ivec2(b.x,a.y),0).rgb,f.x),
               mix(texelFetch(t,ivec2(a.x,b.y),0).rgb,texelFetch(t,b,0).rgb,f.x),f.y);
}
vec3 correctLighting(vec3 src, vec2 uv) {
    float weight = localWeight(uv);
    float gain = u_gain.x*mix(1.0,u_gain.y,weight);
    vec3 halfRgb = bilinear(u_half,uv);
    const vec3 luma = vec3(0.2126,0.7152,0.0722);
    vec3 delta = halfRgb-src;
    float dy = dot(delta,luma);
    float y = dot(src,luma);
    float denoise = min(0.6,(u_gain.x-1.0)*0.45) * (1.0-smoothstep(0.025,0.09,abs(dy)))
                  * (1.0-smoothstep(0.18,0.48,y));
    vec3 chroma = delta-dy;
    denoise *= 1.0-smoothstep(0.065,0.16,max(abs(chroma.r),max(abs(chroma.g),abs(chroma.b))));
    // Shadow luma: the lift multiplies sensor noise by the curve's slope, steepest in the
    // darks. Flat dark pixels move toward the local average; a step well above the noise
    // (an edge) and everything above the shadows keep full resolution.
    float smoothY = clamp((gain-1.0)*0.3,0.0,0.7) * (1.0-smoothstep(0.02,0.06,abs(dy)))
                  * (1.0-smoothstep(0.12,0.40,y));
    // Webcam colour noise comes in blotches (4:2:0 chroma and its compression blocks, 16 px
    // at 720p) that a half-resolution average cannot reach. Where the lift is strong, take the
    // colour of the 1/16-resolution average (a few blocks across) instead, but only where the area is flat in luma
    // at that scale (no outline, no edge) and its colour agrees with the fine average's
    // (real colour structure stays). Luma is untouched; the mean colour, so saturation, stays.
    vec3 wideRgb = bilinear(u_wide,uv);
    vec3 dW = wideRgb-src;
    float dyW = dot(dW,luma);
    vec3 dc = (halfRgb-dot(halfRgb,luma)) - (wideRgb-dyW-dot(src,luma));
    float wideC = clamp((gain-1.0)*WIDE_SLOPE,0.0,WIDE_MAX) * (1.0-smoothstep(WIDE_FLAT_LO,WIDE_FLAT_HI,abs(dyW)))
                * (1.0-smoothstep(WIDE_AGREE_LO,WIDE_AGREE_HI,max(abs(dc.r),max(abs(dc.g),abs(dc.b)))));
    vec3 c = clamp(src + mix(denoise*chroma,dW-dyW,wideC) + smoothY*dy,0.0,1.0);
    vec3 linear = mix(c/12.92,pow((c+0.055)/1.055,vec3(2.4)),greaterThan(c,vec3(0.04045)));
    float lift = mix(1.0,gain,smoothstep(0.0004,0.004,dot(linear,luma)));
    vec3 rgb = linear*vec3(u_gain.z,1,u_gain.w);
    float peak = max(max(rgb.r,rgb.g),rgb.b);
    float scale = lift/(1.0+(lift-1.0)*peak);
    float p = peak*scale;
    float limit = max(1.0,max(u_gain.z,u_gain.w));
    float curvature = (limit-1.0)/(0.2*(limit-0.8));
    float excess = max(0.0,p-0.8);
    if (p>0.8) scale *= (0.8+excess/(1.0+curvature*excess))/p;
    c = rgb*scale;
    return mix(12.92*c,1.055*pow(c,vec3(1.0/2.4))-0.055,greaterThan(c,vec3(0.0031308)));
}
