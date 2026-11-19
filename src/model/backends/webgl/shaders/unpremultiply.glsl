#version 300 es
// Normalized convolution, part 2: rgb / max(a, eps) after blurring (rgb·w, w). Where the
// blur footprint is all subject (a -> 0) it fades to black, which only shows
// where the subject's own alpha is ~1 anyway.
// Elementwise: input(s) and output share shape (4 channels).

precision highp float;

uniform sampler2D u_input;

out vec4 fragColor;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    vec4 t = texelFetch(u_input, fc, 0);
    fragColor = vec4(t.rgb / max(t.a, 2e-3), 1.0);
}
