#version 300 es
// Display-alpha ramp: a = smoothstep(lo, hi, alpha.x). Tightens the soft edge the
// compositor blends over, so less of the original (sharp) room shows through the
// transition around a subject. Applied to the display alpha only (after the
// temporal stabilizer, never fed back into it). alpha in .x; output (a, 0, 0, 0).
// Elementwise: input(s) and output share shape (4 channels).

precision highp float;

uniform sampler2D u_input;
uniform float u_lo;
uniform float u_hi;

out vec4 fragColor;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    float a = texelFetch(u_input, fc, 0).x;
    float t = clamp((a - u_lo) / max(u_hi - u_lo, 1e-4), 0.0, 1.0);
    fragColor = vec4(t * t * (3.0 - 2.0 * t), 0.0, 0.0, 0.0);
}
