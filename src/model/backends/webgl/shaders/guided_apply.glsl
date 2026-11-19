#version 300 es
// Guided upsampler, step 4, at output resolution: boxed (a, b) sampled bilinearly
// (align_corners = false) and applied to the full-resolution luminance,
// alpha = clamp(a·I + b, 0, 1), then optionally smoothstep(lo, hi) (when hi > lo).
// The matte's edges follow the camera's edges at full resolution instead of the
// model's grid, without reading or changing the temporal history.

precision highp float;
precision highp int;

uniform sampler2D u_image;
uniform sampler2D u_coeffs;
uniform int u_out_w;
uniform int u_out_h;
uniform int u_in_w;
uniform int u_in_h;
uniform float u_lo;
uniform float u_hi;

out vec4 fragColor;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    float src_x = (float(fc.x) + 0.5) * float(u_in_w) / float(u_out_w) - 0.5;
    float src_y = (float(fc.y) + 0.5) * float(u_in_h) / float(u_out_h) - 0.5;
    int x0 = int(floor(src_x)), y0 = int(floor(src_y));
    float fx = src_x - float(x0), fy = src_y - float(y0);
    int xa = clamp(x0, 0, u_in_w - 1), xb = clamp(x0 + 1, 0, u_in_w - 1);
    int ya = clamp(y0, 0, u_in_h - 1), yb = clamp(y0 + 1, 0, u_in_h - 1);
    vec2 ab = mix(mix(texelFetch(u_coeffs, ivec2(xa, ya), 0).xy, texelFetch(u_coeffs, ivec2(xb, ya), 0).xy, fx),
                  mix(texelFetch(u_coeffs, ivec2(xa, yb), 0).xy, texelFetch(u_coeffs, ivec2(xb, yb), 0).xy, fx), fy);
    float I = dot(texelFetch(u_image, fc, 0).rgb, vec3(0.299, 0.587, 0.114));
    float a = clamp(ab.x * I + ab.y, 0.0, 1.0);
    if (u_hi > u_lo) {
        float t = clamp((a - u_lo) / (u_hi - u_lo), 0.0, 1.0);
        a = t * t * (3.0 - 2.0 * t);
    }
    fragColor = vec4(a, 0.0, 0.0, 0.0);
}
