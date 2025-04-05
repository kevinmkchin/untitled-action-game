
# Shading in GAME

## Introduction

Targeting OpenGL 4.5 with plans to target Vulkan later. OpenGL 4.3 would cover GPUs from 2012 onwards, but I want to support Vulkan.
Heavily favour the look of global illumination. Pre-computed/baked lighting wherever possible. Video games can't tune lights shot-by-shot as in film. GI tends to result in game environments that look pleasing under a variety of viewing conditions.

## World Lighting

Baked static lightmaps with GPU path-traced lightmapper. The lightmapper bakes global illumination at each lightmap luxel without relying on interpolationg techniques such as irradiance caching.

## Model Lighting
Ground characters in game world and make them seem truly present in their environment.
Allow dynamic models to be convincely lit by pre-computed lighting.

### Irradiance Volume

#### Ambient Cubes

Rather a flat/gray ambient term, use directional ambient term from directional irradiance samples. Directionality gives realistic tint and shading where the underside of a model may receive green bounce light from grass on the ground while the top gets blue from the sky.

The ambient cube represents ambient light flowing through that region of space. It approximates the volume around the cube, like a small environment map of ambient light. The ambient contribution is a simple weighted blending of six colours as a function of world-space normal (at vertex or pixel).

#### Cache visible light sources

Each light probe stores which light sources are visible from it. The most significant light sources (determined by intensity at ideal Lambertian reflectance) are baked into the probe and used to calculate direct lighting for dynamic models.

### Half-Lambert Diffuse

## Screen-space Ambient Occlusion

## High Dynamic Range Rendering

