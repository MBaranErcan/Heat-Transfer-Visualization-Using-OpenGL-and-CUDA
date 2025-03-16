#version 460 core

in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D heatmap;              // [0.0, 1.0]

vec3 tempToColor5C(float temp); // Heat-mapping 5-Color 
vec3 tempToColor2C(float temp); // Heat-mapping 2-Color (Not in use)
vec3 tempToColor3C(float temp); // Heat-mapping 2-Color (Not in use)


void main() {
    float temperature = texture(heatmap, TexCoord).r; // Only using red channel
    vec3 color = tempToColor5C(temperature);
    FragColor = vec4(color, 1.0);
}


//  5-Color Heatmap 
vec3 tempToColor5C(float temp) { 
        temp = clamp(temp, 0.0, 1.0); // clamp value

        vec3 blue = vec3(0.0, 0.0, 1.0);
        vec3 cyan = vec3(0.0, 1.0, 1.0);
        vec3 green = vec3(0.0, 1.0, 0.0);
        vec3 yellow = vec3(1.0, 1.0, 0.0);
        vec3 red = vec3(1.0, 0.0, 0.0);


        vec3 color;
        if (temp < 0.25) {
            color = mix(blue, cyan, temp * 4.0);
        } else if (temp < 0.5) {
            color = mix(cyan, green, (temp - 0.25) * 4.0);
        } else if (temp < 0.75) {
            color = mix(green, yellow, (temp - 0.5) * 4.0);
        } else {
            color = mix(yellow, red, (temp - 0.75) * 4.0);
        }

        return color;
}

// 2-Color Heatmap
vec3 tempToColor2C(float temp) { 
        temp = clamp(temp, 0.0, 1.0); // clamp value

        vec3 blue = vec3(0.0, 0.0, 1.0);
        vec3 red = vec3(1.0, 0.0, 0.0);


        vec3 color;
        return color = mix(blue, red, temp);
}

// 3-Color Heatmap
vec3 tempToColor3C(float temp) { 
        temp = clamp(temp, 0.0, 1.0); // clamp value

        vec3 blue =     vec3(0.0, 0.0, 1.0);
        vec3 white =    vec3(1.0, 1.0, 1.0);
        vec3 red =      vec3(1.0, 0.0, 0.0);


        vec3 color;
        if (temp < 0.50) {
            color = mix(blue, white, temp * 2.0);
        } else {
            color = mix(white, red, (temp - 0.5) * 2.0);
        }

        return color;
}