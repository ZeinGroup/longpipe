#version 300 es
// Guided upsampler, step 1 (fast guided filter, He et al. 2015 / Wu et al. 2018): at the
// model's grid, pack (I, p, I·p, I·I) — I the guide's luminance (the camera frame
// resampled to the model's input), p the matte (.x).

precision highp float;
precision highp int;

uniform sampler2D u_guide;
uniform sampler2D u_alpha;

out vec4 fragColor;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    float I = dot(texelFetch(u_guide, fc, 0).rgb, vec3(0.299, 0.587, 0.114));
    float p = texelFetch(u_alpha, fc, 0).x;
    fragColor = vec4(I, p, I * p, I * I);
}
