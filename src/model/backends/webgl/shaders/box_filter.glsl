#version 300 es
// Guided upsampler, step 2: separable box mean (2·radius + 1 taps, edge replicated)
// along one axis, all four channels.

precision highp float;
precision highp int;

uniform sampler2D u_input;
uniform int u_w;
uniform int u_h;
uniform int u_radius;
uniform int u_vertical;

out vec4 fragColor;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    vec4 s = vec4(0.0);
    for (int k = -u_radius; k <= u_radius; k++) {
        ivec2 q = u_vertical == 1 ? ivec2(fc.x, clamp(fc.y + k, 0, u_h - 1)) : ivec2(clamp(fc.x + k, 0, u_w - 1), fc.y);
        s += texelFetch(u_input, q, 0);
    }
    fragColor = s / float(2 * u_radius + 1);
}
