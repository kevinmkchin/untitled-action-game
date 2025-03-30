
void BindUniformsForModelLighting(GPUShader &Shader, map_info_t &MapInfo, vec3 ModelPosition)
{
    size_t LightCacheIndex = MapInfo.LightCacheVolume->IndexByPosition(ModelPosition);
    lc_ambient_t AmbientCube = MapInfo.LightCacheVolume->AmbientCubes[LightCacheIndex];
    lc_light_indices_t LightIndices = MapInfo.LightCacheVolume->SignificantLightIndices[LightCacheIndex];

    int DoSunLight = 0;
    fixed_array<vec3> PointLightsPos = fixed_array<vec3>(4, MemoryType::Frame);
    fixed_array<float> PointLightsAttLin = fixed_array<float>(4, MemoryType::Frame);
    fixed_array<float> PointLightsAttQuad = fixed_array<float>(4, MemoryType::Frame);
    for (int i = 0; i < 4; ++i)
    {
        short LightIndex = *(((short*)&LightIndices) + i);
        if (LightIndex < 0)
            continue;
        if (LightIndex == SUNLIGHTINDEX)
        {
            DoSunLight = 1;
            continue;
        }

        const static_point_light_t &PointLight = MapInfo.PointLights[LightIndex];
        PointLightsPos.put(PointLight.Pos);
        PointLightsAttLin.put(PointLight.AttenuationLinear);
        PointLightsAttQuad.put(PointLight.AttenuationQuadratic);
    }

    // I could keep global point light and sun light info on the GPU and index into it
    // What about ambient cube? that's a lot of data. Nah, that should be passed as uniform per instance

    i32 loc0 = GetCachedUniformLocation(Shader, "ModelLighting.AmbientCube[0]");
    i32 loc1 = GetCachedUniformLocation(Shader, "ModelLighting.DoSunLight");
    i32 loc2 = GetCachedUniformLocation(Shader, "ModelLighting.DirectionToSun");
    i32 loc3 = GetCachedUniformLocation(Shader, "ModelLighting.PointLightsCount");
    i32 loc4 = GetCachedUniformLocation(Shader, "ModelLighting.PointLightsPos[0]");
    i32 loc5 = GetCachedUniformLocation(Shader, "ModelLighting.PointLightsAttLin[0]");
    i32 loc6 = GetCachedUniformLocation(Shader, "ModelLighting.PointLightsAttQuad[0]");
    glUniform1fv(loc0, 6, (float*)&AmbientCube);
    glUniform1i(loc1, DoSunLight);
    glUniform3fv(loc2, 1, (float*)&MapInfo.DirectionToSun);
    glUniform1i(loc3, PointLightsPos.lenu());
    glUniform3fv(loc4, 4, (float*)PointLightsPos.data);
    glUniform1fv(loc5, 4, (float*)PointLightsAttLin.data);
    glUniform1fv(loc6, 4, (float*)PointLightsAttQuad.data);
}
