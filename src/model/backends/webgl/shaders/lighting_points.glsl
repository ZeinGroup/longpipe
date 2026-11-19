#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_hm, u_box;
out vec4 result;
// Restrict detection to the primary box, so peaks from different faces cannot
// form an eye pair. The five channels are eyes, nose, then mouth corners.
void main() {
    int k = int(gl_FragCoord.x);
    ivec2 size = textureSize(u_hm, 0) / ivec2(2, 1);
    vec4 box = texelFetch(u_box, ivec2(0), 0);
    float side = box.z * float(size.x);
    ivec2 lo = max(ivec2(0), ivec2(box.xy * vec2(size) - side));
    ivec2 hi = min(size, ivec2(box.xy * vec2(size) + side + 1.0));
    float peak = 0.0;
    ivec2 pos = lo;
    for (int y = lo.y; y < hi.y; y++) for (int x = lo.x; x < hi.x; x++) {
        float v = texelFetch(u_hm, ivec2(x*2+k/4,y), 0)[k%4];
        if (v > peak) { peak = v; pos = ivec2(x,y); }
    }
    vec2 sum = vec2(0);
    float mass = 0.0;
    for (int y=-2; y<=2; y++) for (int x=-2; x<=2; x++) {
        ivec2 p = clamp(pos + ivec2(x,y), ivec2(0), size-1);
        float v = texelFetch(u_hm, ivec2(p.x*2+k/4,p.y), 0)[k%4];
        sum += v * (vec2(p) + 0.5); mass += v;
    }
    result = vec4(sum / (max(mass, 1e-6) * vec2(size)), min(box.w,peak), 0);
}
