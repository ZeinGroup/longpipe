#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_alpha, u_points;
uniform int u_w, u_h;
out vec4 result;
float disk(vec2 p, vec2 c, float radius) {
    return 1.0 - smoothstep(0.45 * radius, radius, length(p-c));
}
void main() {
    vec2 grid = vec2(u_w,u_h);
    vec2 uv = gl_FragCoord.xy / grid;
    ivec2 size = textureSize(u_alpha, 0);
    // Interior over two rings. The inner ring erodes the silhouette; the
    // outer ring feathers the lift over ~4% of picture height inside it.
    float inner = 1.0, outer = 0.0;
    for (int y=-1; y<=1; y++) for (int x=-1; x<=1; x++) {
        vec2 d = vec2(x,y) / grid;
        inner = min(inner, texelFetch(u_alpha, clamp(ivec2((uv+1.5*d)*vec2(size)),ivec2(0),size-1),0).x);
        outer += texelFetch(u_alpha, clamp(ivec2((uv+3.0*d)*vec2(size)),ivec2(0),size-1),0).x;
    }
    float person = smoothstep(0.75,0.98,inner) * smoothstep(0.65,1.0,outer/9.0);
    vec3 le = texelFetch(u_points,ivec2(0,0),0).xyz, re = texelFetch(u_points,ivec2(1,0),0).xyz;
    vec3 nose = texelFetch(u_points,ivec2(2,0),0).xyz;
    vec3 lm = texelFetch(u_points,ivec2(3,0),0).xyz, rm = texelFetch(u_points,ivec2(4,0),0).xyz;
    vec2 axis = (re.xy-le.xy)*grid;
    float span = max(length(axis),0.01);
    axis /= span;
    vec2 down = vec2(-axis.y,axis.x);
    vec2 origin = (le.xy+re.xy)*0.5;
    vec2 mouth = ((lm.xy+rm.xy)*0.5-origin)*grid/span;
    down *= sign(dot(mouth,down));
    vec2 delta = (uv-origin)*grid/span;
    vec2 p = vec2(dot(delta,axis), dot(delta,down));
    float height = dot(mouth,down);
    float conf = min(min(le.z,re.z),min(nose.z,min(lm.z,rm.z)));
    conf = smoothstep(0.15,0.30,conf) * smoothstep(1.3,3.0,span)
         * smoothstep(0.45,0.8,height) * (1.0-smoothstep(1.6,2.1,height));
    float cheeks = disk(vec2(abs(p.x),p.y),vec2(0.49,0.58*height),0.34);
    float forehead = disk(p,vec2(0,-0.46),0.40);
    float excluded = max(disk(vec2(abs(p.x),p.y),vec2(0.5,0),0.31),
                         disk((uv-(lm.xy+rm.xy)*0.5)*grid/span,vec2(0),0.5));
    float face = conf * max(cheeks,forehead) * (1.0-excluded);
    float broad = conf * (1.0-smoothstep(0.42,1.22,length(p-vec2(0,0.28))));
    result = vec4(person*(0.65+0.35*broad), person, person*face, 0);
}
