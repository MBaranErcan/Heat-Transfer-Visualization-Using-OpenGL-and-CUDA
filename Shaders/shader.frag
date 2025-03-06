#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D heatmap;

void main() {
    float temperature = texture(heatmap, TexCoord).r; // Only using red channel
    vec3 color = vec3(temperature, temperature * 0.5, 1.0 - temperature); // Blue to Red gradient
    FragColor = vec4(color, 1.0);
}