#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_raw, u_guide;
out vec4 result;
vec3 linearize(vec3 c) {
    return mix(c/12.92, pow((c+0.055)/1.055, vec3(2.4)), greaterThan(c,vec3(0.04045)));
}
void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    ivec2 tile = ivec2(px.x/3,px.y)*6;
    int group = px.x%3;
    vec3 sums = vec3(0), masses = vec3(0);
    vec2 gains = vec2(0);
    float total = 0.0, neutral = 0.0;
    vec2 chromaticity = vec2(0);
    for (int y=0; y<6; y++) for (int x=0; x<6; x++) {
        ivec2 p = tile+ivec2(x,y);
        vec3 encoded = texelFetch(u_raw,p,0).rgb;
        vec3 rgb = linearize(encoded);
        float lum = dot(rgb,vec3(0.2126,0.7152,0.0722));
        float logY = log2(max(lum,0.001));
        total += logY;
        chromaticity += rgb.rb / max(0.015,rgb.r+rgb.g+rgb.b);
        if (lum < 0.004 || max(max(encoded.r,encoded.g),encoded.b) >= 0.975) continue;
        vec4 guide = texelFetch(u_guide,p,0);
        vec3 weights = vec3(1,guide.yz);
        sums += weights*logY; masses += weights;
        vec2 ratios = log2(max(rgb.g,0.001)/max(rgb.rb,vec2(0.001)));
        float chroma = max(abs(ratios.x),abs(ratios.y));
        float w = (1.0-smoothstep(0.23,0.65,chroma)) * smoothstep(0.015,0.055,lum) * (1.0-guide.y);
        gains += ratios*w; neutral += w;
    }
    result = group==0 ? vec4(sums,total/36.0) : group==1 ? vec4(masses,neutral) : vec4(gains,chromaticity/36.0);
}
