#version 300 es
// Background-only blur input (normalized convolution, part 1): the image weighted by the
// background's share w = 1 - alpha, with w itself in .a. Run through the blur
// pyramid and divided back (Unpremultiply), the blurred background takes no
// colour from the subject: no halo of the person around the silhouette.
// Elementwise: input(s) and output share shape (4 channels).

precision highp float;

uniform sampler2D u_image;
uniform sampler2D u_alpha;

out vec4 fragColor;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    float w = 1.0 - clamp(texelFetch(u_alpha, fc, 0).x, 0.0, 1.0);
    fragColor = vec4(texelFetch(u_image, fc, 0).rgb * w, w);
}
