#version 450

layout(std140) uniform UBO
{
    float ubo[4];
} _14;

layout(location = 0) out float FragColor;

void main()
{
    FragColor = _14.ubo[1];
}

