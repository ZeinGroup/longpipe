#version 300 es
// Flow-gated temporal stabilizer. Per pixel:
//   env = max(|flow.xy|, release·envPrev.y)   peak-hold
//   g   = max(clamp((env - tLo)/(tHi - tLo), 0, 1), gDiv, gChg, leak)
//   gChg = clamp((|pred - ref| - tChg)/chgScale, 0, 1), off when chgScale is 0
//   out = vec4((g·pred + (1-g)·ref).x, env, 0, 0)

precision highp float;
precision highp int;

uniform sampler2D u_flow;
uniform sampler2D u_pred;
uniform sampler2D u_ref;
uniform sampler2D u_env_prev;
uniform int   u_w;
uniform int   u_h;
uniform float u_t_lo;
uniform float u_t_hi;
uniform float u_leak;
uniform float u_release;
uniform float u_t_div;
uniform float u_div_scale;
uniform float u_t_chg;
uniform float u_chg_scale;
uniform int   u_step_x;
uniform int   u_step_y;

out vec4 fragColor;

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    float mag     = length(texelFetch(u_flow, ivec2(x, y), 0).xy);
    float envPrev = texelFetch(u_env_prev, ivec2(x, y), 0).y;
    float env     = max(mag, u_release * envPrev);

    // Flow divergence over a ±step finite-difference (clamped to the edges).
    int xr = min(x + u_step_x, u_w - 1);
    int xl = max(x - u_step_x, 0);
    int yd = min(y + u_step_y, u_h - 1);
    int yu = max(y - u_step_y, 0);
    float dfx = texelFetch(u_flow, ivec2(xr, y), 0).x - texelFetch(u_flow, ivec2(xl, y), 0).x;
    float dfy = texelFetch(u_flow, ivec2(x, yd), 0).y - texelFetch(u_flow, ivec2(x, yu), 0).y;
    float divg = abs(dfx + dfy);

    float gMag = clamp((env - u_t_lo) / max(u_t_hi - u_t_lo, 1e-3), 0.0, 1.0);
    float gDiv = clamp((divg - u_t_div) / max(u_div_scale, 1e-3), 0.0, 1.0);
    float pred = texelFetch(u_pred, ivec2(x, y), 0).x;
    float refv = texelFetch(u_ref,  ivec2(x, y), 0).x;
    float gChg = u_chg_scale > 0.0 ? clamp((abs(pred - refv) - u_t_chg) / u_chg_scale, 0.0, 1.0) : 0.0;
    float g = max(max(max(gMag, gDiv), gChg), u_leak);
    float stab = g * pred + (1.0 - g) * refv;

    fragColor = vec4(stab, env, 0.0, 0.0);
}
