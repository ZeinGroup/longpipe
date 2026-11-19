#version 300 es
// Guided upsampler, step 3: per-pixel linear model of the matte in the guide's luminance,
// a = cov(I, p) / (var(I) + eps), b = mean(p) − a·mean(I), from boxed (I, p, I·p, I·I).

precision highp float;
precision highp int;

uniform sampler2D u_input;
uniform float u_eps;

out vec4 fragColor;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    vec4 m = texelFetch(u_input, fc, 0);   // (mean I, mean p, mean I·p, mean I·I)
    float a = (m.z - m.x * m.y) / (max(m.w - m.x * m.x, 0.0) + u_eps);
    fragColor = vec4(a, m.y - a * m.x, 0.0, 0.0);
}
