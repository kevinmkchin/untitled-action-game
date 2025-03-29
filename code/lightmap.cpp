#include "lightmap.h"

#include <radiant.h>

#include "lm_oct.cpp" // only needed by lightmap.cpp


static LevelPolygonOctree LightMapOcclusionTree;
static std::vector<FlatPolygonCollider> MapSurfaceColliders;

static radiant_bake_info_t RadiantBakeInfo;


void lightmapper_t::BakeStaticLighting(game_map_build_data_t& BuildData)
{
    BuildDataShared = &BuildData;

    // TODO(Kevin): get rid of max limit and just dynamically figure out how many texels we need
    arrsetcap(all_lm_pos, MaxNumTexels);
    arrsetcap(all_lm_norm, MaxNumTexels);
    arrsetcap(all_light_global, MaxNumTexels);
    arrsetcap(all_lm_tangent, MaxNumTexels);
    arrsetcap(all_light_direct, MaxNumTexels);
    arrsetcap(all_light_indirect, MaxNumTexels);

    PrepareFaceLightmapsAndTexelStorage();

    PackLightmapsAndMapLocalUVToGlobalUV();

    // lmuvcaches must contain the global lightmap uvs before generating vertices
    GenerateLevelVertices();

    // The Big Lightmap Atlas
    arrsetcap(LIGHTMAPATLAS, lightMapAtlasW*lightMapAtlasH);


    dynamic_array<vec3> WorldGeometryVertices; // World geometry in triangles world positions only
    dynamic_array<radiant_pointlight_t> PointLightInfos;
    for (auto& vbpair : BuildDataShared->VertexBuffers)
    {
        const std::vector<float>& vb = vbpair.second;
        ASSERT(vb.size() % 10 == 0);
        for (size_t i = 0; i < vb.size(); i += 10)
        {
            WorldGeometryVertices.put(vec3(vb[i+0], vb[i+1], vb[i+2]));
        }
    }
    RadiantBakeInfo.OutputLightmap = all_light_global;
    RadiantBakeInfo.OutputLightmapSize = UsedLightmapTexelCount;
    RadiantBakeInfo.LightMapTexelPositions = (radiant_vec3_t*)all_lm_pos;
    RadiantBakeInfo.LightMapTexelNormals = (radiant_vec3_t*)all_lm_norm;
    RadiantBakeInfo.WorldGeometryVertices = (radiant_vec3_t*)WorldGeometryVertices.data;
    RadiantBakeInfo.WorldGeometryVerticesCount = WorldGeometryVertices.lenu();
    for (int i = 0; i < BuildDataShared->PointLights.lenu(); ++i)
    {
        static_point_light_t PLight = BuildDataShared->PointLights[i];
        radiant_pointlight_t PointLightInfo;
        PointLightInfo.Position = *((radiant_vec3_t*)&PLight.Pos);
        PointLightInfo.AttenuationLinear = PLight.AttenuationLinear;
        PointLightInfo.AttenuationQuadratic = PLight.AttenuationQuadratic;
        PointLightInfos.put(PointLightInfo);
    }
    RadiantBakeInfo.PointLights = PointLightInfos.data;
    RadiantBakeInfo.PointLightsCount = BuildDataShared->PointLights.lenu();
    RadiantBakeInfo.DirectionToSun = *((radiant_vec3_t*)&BuildDataShared->DirectionToSun);
    RadiantBakeInfo.SkyboxColor = { 0.53f, 0.81f, 0.92f };
    RadiantBakeInfo.SkyboxBrightness = 0.4f;
    RadiantBakeInfo.NumberOfSampleRaysPerTexel = 4096;
    RadiantBakeInfo.NumberOfLightBounces = 3;
    RadiantBakeInfo.BakeDirectLighting = true;
    RadiantBakeInfo.CacheDirectLightIndices = false;
    RadiantBake(RadiantBakeInfo);


#if 0
    // direct lighting
    GenerateLightmapOcclusionTestTree();

    u32 numpatches = (u32)arrlenu(all_lm_norm);
    ASSERT(numpatches == (u32)arrlenu(all_lm_pos));
    ASSERT(numpatches == (u32)arrlenu(all_light_global));
    u32 progress10 = u32((float)numpatches * 0.1f);
    u32 progress20 = u32((float)numpatches * 0.2f);
    u32 progress30 = u32((float)numpatches * 0.3f);
    u32 progress40 = u32((float)numpatches * 0.4f);
    u32 progress50 = u32((float)numpatches * 0.5f);
    u32 progress60 = u32((float)numpatches * 0.6f);
    u32 progress70 = u32((float)numpatches * 0.7f);
    u32 progress80 = u32((float)numpatches * 0.8f);
    u32 progress90 = u32((float)numpatches * 0.9f);
    std::thread t0 = std::thread(&lightmapper_t::ThreadSafe_DoDirectLightingIntoLightMap, this, 0, progress10);
    std::thread t1 = std::thread(&lightmapper_t::ThreadSafe_DoDirectLightingIntoLightMap, this, progress10, progress20);
    std::thread t2 = std::thread(&lightmapper_t::ThreadSafe_DoDirectLightingIntoLightMap, this, progress20, progress30);
    std::thread t3 = std::thread(&lightmapper_t::ThreadSafe_DoDirectLightingIntoLightMap, this, progress30, progress40);
    std::thread t4 = std::thread(&lightmapper_t::ThreadSafe_DoDirectLightingIntoLightMap, this, progress40, progress50);
    std::thread t5 = std::thread(&lightmapper_t::ThreadSafe_DoDirectLightingIntoLightMap, this, progress50, progress60);
    std::thread t6 = std::thread(&lightmapper_t::ThreadSafe_DoDirectLightingIntoLightMap, this, progress60, progress70);
    std::thread t7 = std::thread(&lightmapper_t::ThreadSafe_DoDirectLightingIntoLightMap, this, progress70, progress80);
    std::thread t8 = std::thread(&lightmapper_t::ThreadSafe_DoDirectLightingIntoLightMap, this, progress80, progress90);
    std::thread t9 = std::thread(&lightmapper_t::ThreadSafe_DoDirectLightingIntoLightMap, this, progress90, numpatches);

    // indirect lighting

    std::unordered_map<u32, std::vector<float>>& VertexBuffers = BuildDataShared->VertexBuffers;
    float *patches_vb = NULL; 
    for (auto& vbpair : VertexBuffers)
    {
        const std::vector<float>& vb = vbpair.second;
        float *writeto = arraddnptr(patches_vb, vb.size());
        memcpy(writeto, vb.data(), sizeof(float)*vb.size());
    }
    CreateFaceBatch(&SceneLightingModel);
    RebindFaceBatch(&SceneLightingModel, sizeof(float)*arrlenu(patches_vb), patches_vb);
    arrfree(patches_vb);
    SceneLightingModel.ColorTexture = Assets.DefaultMissingTexture;


    // DECLARE LIGHT MAP ATLAS
    float *LIGHT_MAP_ATLAS = NULL;
    arrsetcap(LIGHT_MAP_ATLAS, lightMapAtlasW*lightMapAtlasH);

    CreateGPUTextureFromBitmap(&SceneLightingModel.LightMapTexture, (void*)LIGHT_MAP_ATLAS, 
        lightMapAtlasW, lightMapAtlasH, GL_R32F, GL_RED, GL_LINEAR, GL_LINEAR, GL_FLOAT);


    GPUFrameBuffer HemicubeFBO; // the cube faces are laid out horizontally
    HemicubeFBO.width = HemicubeFaceW*5;
    HemicubeFBO.height = HemicubeFaceH;
    CreateGPUFrameBuffer(&HemicubeFBO, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    u32 HemicubePBO;
    glGenBuffers(1, &HemicubePBO);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, HemicubePBO);
    const GLsizeiptr NumFloatsPerHemicubeFace = HemicubeFaceW * HemicubeFaceH * 4;
    glBufferData(GL_PIXEL_PACK_BUFFER, 5*NumFloatsPerHemicubeFace*sizeof(float), NULL, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    // NOTE(Kevin 2025-01-27): This is tricky. I must disable GL_BLEND here because I'm currently
    // using the alpha channel of the hemicubes as a tag for whether the texel is a backface or not.
    // Without explictly disabling, hemicube rendering won't work properly after rendering the level
    // editor because it enables blending there.
    glDisable(GL_BLEND); 
    UseShader(PatchesIDShader);

    float HEMICUBE_NEARCLIP = 1.f;
    mat4 HemicubePerspectiveMatrix = ProjectionMatrixPerspective(90.f*GM_DEG2RAD, 1.f, HEMICUBE_NEARCLIP, GAMEPROJECTION_FARCLIP);
    GLBindMatrix4fv(PatchesIDShader, "projMatrix", 1, HemicubePerspectiveMatrix.ptr());

    glBindFramebuffer(GL_FRAMEBUFFER, HemicubeFBO.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0); // Asynchronously read pixel data into the PBO
    glBindBuffer(GL_PIXEL_PACK_BUFFER, HemicubePBO);

    CreateMultiplierMap();

    t0.join();
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    t7.join();
    t8.join();
    t9.join();

#define LM_BACKFACE_INDICATOR 0.69f
// #define LM_MARK_BAD_TEXEL -0.05f

    for (int bounces = 0; bounces < 2; ++bounces)
    {
        // Copy the radiance info thus far
        // Would be just direct lighting for first bounce, but future bounces
        // would be direct light + accumulated indirect light.
        for (size_t i = 0; i < PackedLMRects.lenu(); ++i)
        {
            stbrp_rect rect = PackedLMRects[i];
            if (rect.was_packed == 0) continue;
            ASSERT(rect.was_packed != 0);
            const lm_face_t& lmface = FaceLightmaps[rect.id];
            BlitRect((u8*)LIGHT_MAP_ATLAS, lightMapAtlasW, lightMapAtlasH, 
                (u8*)lmface.light, lmface.w, lmface.h, rect.x, rect.y, sizeof(float));
        }
        // Update SceneLightingModel to use the updated radiance information
        UpdateGPUTextureFromBitmap(&SceneLightingModel.LightMapTexture, (u8*)LIGHT_MAP_ATLAS,
            lightMapAtlasW, lightMapAtlasH);


        for (int FaceIndex = 0; FaceIndex < (int)PackedLMRects.lenu(); ++FaceIndex)
        {
            // TODO reset "cache" - just an array for now

            const lm_face_t& FaceLightmap = FaceLightmaps[FaceIndex];
            u32 NumTexelsOnFace = FaceLightmap.w * FaceLightmap.h;
            for (u32 TexelOffset = 0; TexelOffset < NumTexelsOnFace; ++TexelOffset)
            {
                CalcBounceLightForTexel(FaceLightmap, TexelOffset, NumFloatsPerHemicubeFace);
            }

            // some post processing?

            // // TODO Irradiance Cache population is complete for this face
            // //      Now go through every texel of this face again and perform interpolation!
            // //      Unless the texel was a hemicube sampled texel (exact irradiance known)
            // //      need to communicate that somehow
            // // set just the all_light_indirect buffer
            // for (u32 i = 0; i < NumTexelsOnFace; ++i)
            // {
            //     // TODO check if texel should be sampled or ignored
            //     // if (ignore) 
            //     //     continue;

            //     // if should be sampled, then it must be covered by at least one cache entry
            //     // if it is not covered, then something went wrong!!!

            //     float IrradianceAtTexel = 0.f;
            //     FaceLightmap.light_indirect[i] = IrradianceAtTexel;
            // }
            // // At this point, all_light_indirect for the texels we care about have been updated
        }

        // Update global illumination buffer
        u32 all_lmsz = (u32)arrlenu(all_light_global);
        for (u32 i = 0; i < all_lmsz; ++i)
        {
            all_light_global[i] = all_light_direct[i] + all_light_indirect[i];
            // all_light_global[i] = GM_clamp(all_light_global[i], 0.f, 1.f);
        }

        LogMessage("Finished bounce %d", bounces + 1);
    }
    // At this point, global illumination computation is complete

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteBuffers(1, &HemicubePBO);
    DeleteGPUFrameBuffer(&HemicubeFBO);
    DeleteGPUTexture(&SceneLightingModel.LightMapTexture);
    DeleteFaceBatch(&SceneLightingModel);
#endif

    // Final blit to light map atlas
    i32 TotalUsedTexelsInLightmapAtlas = 0;
    for (size_t i = 0; i < PackedLMRects.lenu(); ++i)
    {
        stbrp_rect rect = PackedLMRects[i];
        if (rect.was_packed == 0) continue;
        ASSERT(rect.was_packed != 0);
        const lm_face_t& lmface = FaceLightmaps[rect.id];
        BlitRect((u8*)LIGHTMAPATLAS, lightMapAtlasW, lightMapAtlasH, 
            (u8*)lmface.light, lmface.w, lmface.h, rect.x, rect.y, sizeof(float));
        TotalUsedTexelsInLightmapAtlas += lmface.w * lmface.h;
    }


    LogMessage("- Baking lightmap finished");
    size_t NumPatches = arrlenu(all_lm_pos);
    size_t CapAllPatchData = arrcap(all_lm_pos);
    size_t TotalAllocMemForPatchData
        = sizeof(vec3)*arrcap(all_lm_pos)
        + sizeof(vec3)*arrcap(all_lm_norm)
        + sizeof(vec3)*arrcap(all_lm_tangent)
        + sizeof(float)*arrcap(all_light_global)
        + sizeof(float)*arrcap(all_light_direct)
        + sizeof(float)*arrcap(all_light_indirect);
    LogMessage("      Filled %.1f%% (%zd/%zd) of allocated memory (%zd KB) lightmap baking intermediate data.", 
        float(NumPatches)/float(CapAllPatchData)*100.f, NumPatches, CapAllPatchData,
        TotalAllocMemForPatchData/1000);
    LogMessage("      Filled %.1f%% (%d/%d) of output lightmap texture (%dx%d).",
        float(TotalUsedTexelsInLightmapAtlas)/float(lightMapAtlasW*lightMapAtlasH)*100.f,
        TotalUsedTexelsInLightmapAtlas, lightMapAtlasW*lightMapAtlasH,
        lightMapAtlasW, lightMapAtlasH);

    // MapSurfaceColliders.clear();
    // LightMapOcclusionTree.TearDown();

    FaceLightmaps.free();
    PackedLMRects.free();
    arrfree(all_lm_pos);
    arrfree(all_lm_norm);
    arrfree(all_lm_tangent);
    arrfree(all_light_global);
    arrfree(all_light_direct);
    arrfree(all_light_indirect);

    BuildDataShared = nullptr;
}

void lightmapper_t::GetLightmap(float **PtrToLightMapAtlas, i32 *AtlasWidth, i32 *AtlasHeight)
{
    // Why tf is the lighting data in float not u8? I remember specifically changing it but why...
    // I think I did it so that lighting value can exceed 1.0? 11c6bf455d6e43af48df397945a87fc54baf3759
    // NOTE(Kevin) 2025-03-14: HDR (brightness not clamped to 1.0), apparently reduced color banding
    *PtrToLightMapAtlas = LIGHTMAPATLAS;
    *AtlasWidth = lightMapAtlasW;
    *AtlasHeight = lightMapAtlasH;
}

void lightmapper_t::FreeLightmap()
{
    arrfree(LIGHTMAPATLAS);
}

void lightmapper_t::GenerateLightmapOcclusionTestTree()
{
    std::vector<vec3>& ColliderWorldPoints = BuildDataShared->ColliderWorldPoints;
    std::vector<u32>& ColliderSpans = BuildDataShared->ColliderSpans;

    Bounds MapBounds = Bounds(vec3(-0.17f, -0.17f, -0.17f), vec3(8000, 8000, 8000));
    LightMapOcclusionTree = LevelPolygonOctree(MapBounds, 100, 24);
    MapSurfaceColliders.resize(ColliderSpans.size());
    int iter = 0;
    // later, when only some surfaces have colliders, can't use ColliderSpan, need to traverse all faces again
    for (u32 i = 0; i < ColliderSpans.size(); ++i)
    {
        u32 span = ColliderSpans[i];
        FlatPolygonCollider& surface = MapSurfaceColliders[i];
        surface.pointCloudPtr = &ColliderWorldPoints[iter];
        surface.pointCount = span;
        iter += span;
        LightMapOcclusionTree.Insert(&surface);
    }
}

void lightmapper_t::PrepareFaceLightmapsAndTexelStorage()
{
    ASSERT(!FaceLightmaps.data);
    FaceLightmaps.setcap(BuildDataShared->TotalFaceCount);

    UsedLightmapTexelCount = 0;
    for (int i = 0; i < BuildDataShared->TotalFaceCount; ++i)
    {
        // calculate bounds, and divide into patches/texels
        MapEdit::Face *face = MapEdit::LevelEditorFaces[i];

        vec3 v0 = face->loopbase->v->pos;
        vec3 v1 = face->loopbase->loopNext->v->pos;
        vec3 v2 = face->loopbase->loopNext->loopNext->v->pos;
        vec3 fn = Normalize(Cross(v1-v0, v2-v0));
        vec3 basisU = Normalize(v1-v0);
        vec3 basisV = Normalize(Cross(fn, basisU));

        vec2 minuv = vec2(FLT_MAX, FLT_MAX);
        vec2 maxuv = vec2(-FLT_MAX, -FLT_MAX);
        std::vector<MapEdit::Loop*> loopcycle = face->GetLoopCycle();

        for (MapEdit::Loop *loop : loopcycle)
        {
            vec3 p = loop->v->pos;
            // Project 3D verts unto 2D plane using basis vectors U and V.
            // Relative to v0.
            float u = Dot(p-v0, basisU);
            float v = Dot(p-v0, basisV);
            minuv.x = fminf(minuv.x, u);
            minuv.y = fminf(minuv.y, v);
            maxuv.x = fmaxf(maxuv.x, u);
            maxuv.y = fmaxf(maxuv.y, v);
            vec2 basisUV = vec2(u,v);
            loop->lmuvcache = basisUV; // store basisUV for processing
        }
        
        // Adding padding texel around light map for bilinear filtering. 
        minuv -= vec2(LightMapTexelSize, LightMapTexelSize);
        maxuv += vec2(LightMapTexelSize, LightMapTexelSize);
        vec2 dim = maxuv - minuv;
        dim.x = ceilf(dim.x / LightMapTexelSize) * LightMapTexelSize;
        dim.y = ceilf(dim.y / LightMapTexelSize) * LightMapTexelSize;

        // lm uv
        for (MapEdit::Loop *loop : loopcycle)
        {
            // map lm uv to a local [0,1] value
            // after packing lm rects into larger light map atlas, these will be remapped
            loop->lmuvcache -= minuv;
            loop->lmuvcache.x /= dim.x;
            loop->lmuvcache.y /= dim.y;
            ASSERT(0.f <= loop->lmuvcache.x && loop->lmuvcache.x <= 1.f);
            ASSERT(0.f <= loop->lmuvcache.y && loop->lmuvcache.y <= 1.f);
            // at this point, lmuvcache stores the local uv
        }

        // lm data
        lm_face_t lm;
        lm.faceRef = face;
        lm.w = (i32)(dim.x / LightMapTexelSize);
        lm.h = (i32)(dim.y / LightMapTexelSize);
        i32 lmsz = lm.w*lm.h;
        UsedLightmapTexelCount += lmsz;
        lm.pos = arraddnptr(all_lm_pos, lmsz);
        lm.norm = arraddnptr(all_lm_norm, lmsz);
        lm.light = arraddnptr(all_light_global, lmsz);
        lm.tangent = arraddnptr(all_lm_tangent, lmsz);
        lm.light_direct = arraddnptr(all_light_direct, lmsz);
        lm.light_indirect = arraddnptr(all_light_indirect, lmsz);
        for (i32 pi = 0; pi < lmsz; ++pi)
        {
            i32 x = pi % lm.w;
            i32 y = pi / lm.w;
            float uSampleCenter = minuv.x + LightMapTexelSize*x + LightMapTexelSize*0.5f;
            float vSampleCenter = minuv.y + LightMapTexelSize*y + LightMapTexelSize*0.5f;
            // reverse the projection
            vec3 p = v0 + uSampleCenter * basisU + vSampleCenter * basisV;
            lm.pos[pi] = p;
            lm.norm[pi] = fn;
            lm.tangent[pi] = basisV;
            lm.light[pi] = 0.f;
            lm.light_direct[pi] = 0.f;
            lm.light_indirect[pi] = 0.f;
        }

        FaceLightmaps.put(lm);
    }
}

void lightmapper_t::PackLightmapsAndMapLocalUVToGlobalUV()
{
    ASSERT(!PackedLMRects.data);

    // Pack light maps
    for (size_t i = 0; i < FaceLightmaps.lenu(); ++i)
    {
        stbrp_rect rect;
        rect.id = (int)i;
        rect.w = FaceLightmaps[i].w;
        rect.h = FaceLightmaps[i].h;
        PackedLMRects.put(rect);
    }
    stbrp_node *LMPackerNodes = NULL;
    arrsetlen(LMPackerNodes, lightMapAtlasW);
    stbrp_context LightMapPacker;
    stbrp_init_target(&LightMapPacker, lightMapAtlasW, lightMapAtlasH, LMPackerNodes, (int)arrlenu(LMPackerNodes));
    stbrp_pack_rects(&LightMapPacker, PackedLMRects.data, (int)PackedLMRects.lenu());
    arrfree(LMPackerNodes);
    for (size_t i = 0; i < PackedLMRects.lenu(); ++i)
    {
        stbrp_rect rect = PackedLMRects[i];
        if (rect.was_packed == 0) continue;
        ASSERT(rect.was_packed != 0); // TODO(Kevin): additional light map atlases if couldn't fit into one

        vec2 minuv = vec2((float)(rect.x) / (float)lightMapAtlasW, (float)(rect.y) / (float)lightMapAtlasH);
        vec2 maxuv = vec2((float)(rect.x + rect.w) / (float)lightMapAtlasW, (float)(rect.y + rect.h) / (float)lightMapAtlasH);

        MapEdit::Face *face = FaceLightmaps[rect.id].faceRef;
        std::vector<MapEdit::Loop*> loopcycle = face->GetLoopCycle();
        for (MapEdit::Loop *loop : loopcycle)
        {
            // Map lm uv from local to global in light map atlas
            loop->lmuvcache.x = Lerp(minuv.x, maxuv.x, loop->lmuvcache.x); 
            loop->lmuvcache.y = Lerp(minuv.y, maxuv.y, loop->lmuvcache.y); 
        }
    }
}

void lightmapper_t::GenerateLevelVertices()
{
    // Sort faces by their textures and generate vertex buffers
    std::unordered_map<u32, std::vector<float>>& VertexBuffers = BuildDataShared->VertexBuffers;
    for (size_t i = 0; i < FaceLightmaps.lenu(); ++i)
    {
        MapEdit::Face *face = FaceLightmaps[i].faceRef;
        db_tex_t tex = face->texture;
        if (VertexBuffers.find(tex.persistId) == VertexBuffers.end())
        {
            VertexBuffers.emplace(tex.persistId, std::vector<float>());
        }

        std::vector<float>& vb = VertexBuffers.at(tex.persistId);
        TriangulateFace_ForFaceBatch_QuickDumb(*face, &vb);
    }
}

void lightmapper_t::CreateMultiplierMap()
{
    // this hemicube texels multiplier map only needs to be created once

    float MultiplierBeforeNormalizeSum = 0.f;
    for (int p = 0; p < HemicubeFaceArea; ++p)
    {
        // +0.5f for center of pixel
        float XNorm = (float(p / HemicubeFaceW) + 0.5f) / float(HemicubeFaceHHalf) - 1.f;
        float ZNorm = (float(p % HemicubeFaceW) + 0.5f) / float(HemicubeFaceWHalf) - 1.f;
        float RSquared = XNorm*XNorm + ZNorm*ZNorm + 1.f;

        // Since X and Z are normalized to [-1,1], distance in Y from patch to face is 1
        vec3 PatchToPixelDir = Normalize(vec3(XNorm, 1.f, ZNorm));
        vec3 CameraDirection = vec3(0.f, 1.f, 0.f);
        float CosTheta = Dot(PatchToPixelDir, CameraDirection);

        // For Top face, the surface normal is equal to CameraDirection. 
        // Therefore, Lambert's Cosine Law gives the same result as CosTheta 
        // I can just do CosTheta * CosTheta

        // NOTE(Kevin): Ignacio Castano does SolidAngleOfPixel * CosTheta(for Lambert's law)
        // That's the multiplier for the incoming radiance at that pixel
        // See https://www.ludicon.com/castano/blog/articles/irradiance-caching-part-1/

        float SolidAngleSubtendedByPixel = CosTheta/RSquared;
        // NOTE(Kevin): CosTheta=1/r, so can simplify to SolidAngle = CosTheta^3
        float MultiplierBeforeNormalize = SolidAngleSubtendedByPixel * CosTheta;
        MultiplierBeforeNormalizeSum += MultiplierBeforeNormalize;

        MultiplierMapTop[p] = MultiplierBeforeNormalize;
    }
    for (int p = HemicubeFaceAreaHalf; p < HemicubeFaceArea; ++p)
    {
        float YNorm = (float(p / HemicubeFaceW) + 0.5f) / float(HemicubeFaceHHalf) - 1.f;
        float ZNorm = (float(p % HemicubeFaceW) + 0.5f) / float(HemicubeFaceWHalf) - 1.f;
        float RSquared = YNorm*YNorm + ZNorm*ZNorm + 1.f;

        vec3 PatchToPixelDir = Normalize(vec3(1.f, YNorm, ZNorm));
        vec3 CameraDirection = vec3(1.f, 0.f, 0.f);
        float CosTheta = Dot(PatchToPixelDir, CameraDirection);
        vec3 SurfaceNormal = vec3(0.f, 1.f, 0.f);
        float LambertsCosineLaw = Dot(SurfaceNormal, PatchToPixelDir);

        float SolidAngleSubtendedByPixel = CosTheta/RSquared;
        float MultiplierBeforeNormalize = SolidAngleSubtendedByPixel * LambertsCosineLaw;
        MultiplierBeforeNormalizeSum += MultiplierBeforeNormalize * 4.f; // x4 cuz 4 side faces

        MultiplierMapSide[p-HemicubeFaceAreaHalf] = MultiplierBeforeNormalize;
    }
    // Normalize MultiplierMap such that the sum of all pixels add up to 1
    for (int p = 0; p < HemicubeFaceArea; ++p)
        MultiplierMapTop[p] /= MultiplierBeforeNormalizeSum;
    for (int p = HemicubeFaceAreaHalf; p < HemicubeFaceArea; ++p)
        MultiplierMapSide[p-HemicubeFaceAreaHalf] /= MultiplierBeforeNormalizeSum;
}

#if 0
void lightmapper_t::CalcBounceLightForTexel(const lm_face_t& FaceLightmap, 
    u32 TexelOffset, const GLsizeiptr NumFloatsPerHemicubeFace)
{
    // TODO larger hemicube atlas for download rather than one hemicube by one 

    // if (FaceIndex == 1 && TexelOffset >= 50 && TexelOffset < 52)
    // {
    //     if (RDOCAPI) RDOCAPI->StartFrameCapture(NULL, NULL);
    // }
    
    vec3 patch_i_pos = *(FaceLightmap.pos + TexelOffset); // rename to texel
    vec3 patch_i_normal = *(FaceLightmap.norm + TexelOffset);
    vec3 patch_i_basisV = *(FaceLightmap.tangent + TexelOffset);

    // TODO check if texel should be sampled - this should be computed at texel-creation
    //      time by checking distance from polygon and coverage. This is an optimization step. 
    //      if yes, keep going
    //      if no, continue to next texel

    // TODO check if texel position is covered by any existing irradiance cache entries
    //      if yes, continue to next texel
    //      if no, continue to hemicube sampling

    // Random rotation of hemicube trades banding artifacts for noise
    // banding artifacts is hard to get rid of with just increased hemicube resolution
    // patch_i_basisV = RotateVector(patch_i_basisV, quat((float)rand()/GM_TWOPI, patch_i_normal));
    vec3 patch_i_basisU = Normalize(Cross(patch_i_basisV, patch_i_normal));

    glClearColor(0.f,0.f,0.f,1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, HemicubeFaceW, HemicubeFaceH);
    mat4 HemicubeViewMatrix = ViewMatrixLookAt(patch_i_pos, patch_i_pos + patch_i_normal, patch_i_basisV);
    GLBindMatrix4fv(PatchesIDShader, "viewMatrix", 1, HemicubeViewMatrix.ptr());
    RenderFaceBatch(&PatchesIDShader, &SceneLightingModel);

    // if top face is "front", then 
    // up face
    glViewport(HemicubeFaceW*1, 0, HemicubeFaceW, HemicubeFaceH);
    HemicubeViewMatrix = ViewMatrixLookAt(patch_i_pos, patch_i_pos + patch_i_basisV, patch_i_normal);
    GLBindMatrix4fv(PatchesIDShader, "viewMatrix", 1, HemicubeViewMatrix.ptr());
    RenderFaceBatch(&PatchesIDShader, &SceneLightingModel);
    // down face
    glViewport(HemicubeFaceW*2, 0, HemicubeFaceW, HemicubeFaceH);
    HemicubeViewMatrix = ViewMatrixLookAt(patch_i_pos, patch_i_pos - patch_i_basisV, patch_i_normal);
    GLBindMatrix4fv(PatchesIDShader, "viewMatrix", 1, HemicubeViewMatrix.ptr());
    RenderFaceBatch(&PatchesIDShader, &SceneLightingModel);
    // left face
    glViewport(HemicubeFaceW*3, 0, HemicubeFaceW, HemicubeFaceH);
    HemicubeViewMatrix = ViewMatrixLookAt(patch_i_pos, patch_i_pos + patch_i_basisU, patch_i_normal);
    GLBindMatrix4fv(PatchesIDShader, "viewMatrix", 1, HemicubeViewMatrix.ptr());
    RenderFaceBatch(&PatchesIDShader, &SceneLightingModel);
    // right face
    glViewport(HemicubeFaceW*4, 0, HemicubeFaceW, HemicubeFaceH);
    HemicubeViewMatrix = ViewMatrixLookAt(patch_i_pos, patch_i_pos - patch_i_basisU, patch_i_normal);
    GLBindMatrix4fv(PatchesIDShader, "viewMatrix", 1, HemicubeViewMatrix.ptr());
    RenderFaceBatch(&PatchesIDShader, &SceneLightingModel);

    // if (RDOCAPI) RDOCAPI->EndFrameCapture(NULL, NULL);

    // Putting the glReadPixels together at the end of the draw calls is somehow appreciably faster
    glReadPixels(0, 0, HemicubeFaceW, HemicubeFaceH, GL_RGBA, GL_FLOAT, 0);
    glReadPixels(HemicubeFaceW*1, 0, HemicubeFaceW, HemicubeFaceH, GL_RGBA, GL_FLOAT, (void*)(NumFloatsPerHemicubeFace*sizeof(float)*1));
    glReadPixels(HemicubeFaceW*2, 0, HemicubeFaceW, HemicubeFaceH, GL_RGBA, GL_FLOAT, (void*)(NumFloatsPerHemicubeFace*sizeof(float)*2));
    glReadPixels(HemicubeFaceW*3, 0, HemicubeFaceW, HemicubeFaceH, GL_RGBA, GL_FLOAT, (void*)(NumFloatsPerHemicubeFace*sizeof(float)*3));
    glReadPixels(HemicubeFaceW*4, 0, HemicubeFaceW, HemicubeFaceH, GL_RGBA, GL_FLOAT, (void*)(NumFloatsPerHemicubeFace*sizeof(float)*4));


    // Map the PBO to access pixel data in system memory
    float *FrontFaceData = (float*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY); 
    ASSERT(FrontFaceData);
    float *UpFaceData = FrontFaceData + NumFloatsPerHemicubeFace;
    float *DownFaceData = FrontFaceData + NumFloatsPerHemicubeFace*2;
    float *LeftFaceData = FrontFaceData + NumFloatsPerHemicubeFace*3;
    float *RightFaceData = FrontFaceData + NumFloatsPerHemicubeFace*4;


    int BackfacePixelCount = 0;
    const int BackfaceTolerance = (HemicubeFaceArea + HemicubeFaceAreaHalf * 4) / 10;
    float RadiositiesAccumulator = 0.f;

    for (int p = 0; p < HemicubeFaceArea; ++p)
    {
        vec4 HemicubePixel = { FrontFaceData[p*4], FrontFaceData[p*4+1], 
            FrontFaceData[p*4+2], FrontFaceData[p*4+3] };
        // NOTE(Kevin): radiance is white light only for now (single channel)
        float Radiance = HemicubePixel.x;
        float DifferentialFormFactor = MultiplierMapTop[p];
        RadiositiesAccumulator += DifferentialFormFactor * Radiance;

        if (HemicubePixel.w == LM_BACKFACE_INDICATOR)
            ++BackfacePixelCount;

        // Side faces
        if (p < HemicubeFaceAreaHalf)
            continue;

        HemicubePixel = { UpFaceData[p*4], UpFaceData[p*4+1], 
            UpFaceData[p*4+2], UpFaceData[p*4+3] };
        Radiance = HemicubePixel.x;
        DifferentialFormFactor = MultiplierMapSide[p-HemicubeFaceAreaHalf];
        RadiositiesAccumulator += DifferentialFormFactor * Radiance;

        if (HemicubePixel.w == LM_BACKFACE_INDICATOR)
            ++BackfacePixelCount;

        HemicubePixel = { DownFaceData[p*4], DownFaceData[p*4+1], 
            DownFaceData[p*4+2], DownFaceData[p*4+3] };
        Radiance = HemicubePixel.x;
        DifferentialFormFactor = MultiplierMapSide[p-HemicubeFaceAreaHalf];
        RadiositiesAccumulator += DifferentialFormFactor * Radiance;

        if (HemicubePixel.w == LM_BACKFACE_INDICATOR)
            ++BackfacePixelCount;

        HemicubePixel = { LeftFaceData[p*4], LeftFaceData[p*4+1], 
            LeftFaceData[p*4+2], LeftFaceData[p*4+3] };
        Radiance = HemicubePixel.x;
        DifferentialFormFactor = MultiplierMapSide[p-HemicubeFaceAreaHalf];
        RadiositiesAccumulator += DifferentialFormFactor * Radiance;

        if (HemicubePixel.w == LM_BACKFACE_INDICATOR)
            ++BackfacePixelCount;

        HemicubePixel = { RightFaceData[p*4], RightFaceData[p*4+1], 
            RightFaceData[p*4+2], RightFaceData[p*4+3] };
        Radiance = HemicubePixel.x;
        DifferentialFormFactor = MultiplierMapSide[p-HemicubeFaceAreaHalf];
        RadiositiesAccumulator += DifferentialFormFactor * Radiance;
        
        if (HemicubePixel.w == LM_BACKFACE_INDICATOR)
            ++BackfacePixelCount;

        if (BackfacePixelCount > BackfaceTolerance)
            break;
    }

    if (BackfacePixelCount > BackfaceTolerance)
    {
        // TODO then what do I do about this texel?
        RadiositiesAccumulator = 0.f;
        // RadiositiesAccumulator = LM_MARK_BAD_TEXEL;
    }

    *(FaceLightmap.light_indirect + TexelOffset) = RadiositiesAccumulator;

    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
}
#endif

void lightmapper_t::ThreadSafe_DoDirectLightingIntoLightMap(u32 patchIndexStart, u32 patchIndexEnd)
{
    // calculate direct lighting for patchIndexStart upto and including patchIndexEnd-1

    // 4x multisampling with sparse regular grid distribution
    //  ____ 
    // | *  |
    // |   *|
    // |*   |
    // |  * |
    //  ----

    bool SunExists = BuildDataShared->DirectionToSun != vec3();
    vec3 DLightIncidenceRay = Normalize(BuildDataShared->DirectionToSun);

    for (u32 i = patchIndexStart; i < patchIndexEnd; ++i)
    {
        vec3 patch_position = all_lm_pos[i];
        vec3 patch_normal = all_lm_norm[i];
        vec3 patch_basisV = all_lm_tangent[i];
        vec3 patch_basisU = Normalize(Cross(patch_basisV, patch_normal));

        const float HalfLightMapPatchSz = LightMapTexelSize * 0.5f;
        vec3 SamplePositions[4];
        SamplePositions[0] = patch_position
            + patch_basisU * 0.75f * HalfLightMapPatchSz + patch_basisV * 0.25f * HalfLightMapPatchSz;
        SamplePositions[1] = patch_position
            + patch_basisU * -0.25f * HalfLightMapPatchSz + patch_basisV * 0.75f * HalfLightMapPatchSz;
        SamplePositions[2] = patch_position
            + patch_basisU * -0.75f * HalfLightMapPatchSz + patch_basisV * -0.25f * HalfLightMapPatchSz;
        SamplePositions[3] = patch_position
            + patch_basisU * 0.25f * HalfLightMapPatchSz + patch_basisV * -0.75f * HalfLightMapPatchSz;

        float SampleIntensityAccumulator = 0.f;

        for (u32 sample = 0; sample < 4; ++sample)
        {
            if (SunExists)
            {
                float costheta = Dot(DLightIncidenceRay, patch_normal);
                if (costheta > 0.f)
                {
                    // occlusion test
                    LineCollider ray_collider;
                    ray_collider.a = patch_position + DLightIncidenceRay * 32000.0f;
                    ray_collider.b = patch_position + DLightIncidenceRay * 0.2f;
                    bool occluded = false;

                    if (LightMapOcclusionTree.Query(ray_collider))
                    {
                        occluded = true;
                    }

                    if (!occluded)
                    {
                        float intensity = costheta;
                        intensity = GM_min(intensity, 1.0f);
                        SampleIntensityAccumulator += intensity;
                    }
                }
            }

            for (size_t i = 0; i < BuildDataShared->PointLights.lenu(); ++i)
            {
                static_point_light_t PointLight = BuildDataShared->PointLights[i];

                vec3 incidence_ray = PointLight.Pos - SamplePositions[sample];
                float costheta = Dot(Normalize(incidence_ray), patch_normal);
                if (costheta > 0.f)
                {
                    float dist = Magnitude(incidence_ray);

                    // occlusion test
                    LineCollider ray_collider;
                    ray_collider.a = PointLight.Pos;
                    ray_collider.b = PointLight.Pos - incidence_ray * 0.98f;
                    bool occluded = false;

                    if (LightMapOcclusionTree.Query(ray_collider))
                    {
                        occluded = true;
                    }

                    if (!occluded)
                    {
                        // point light attenuation
                        // doing quadratic component is too dark when there is no gamma correction
                        float atten_lin = 0.02f;
                        float atten_quad = 0.00019f;
                        float attenuation = 1.f / 
                            (1.f + atten_lin * dist + atten_quad * dist * dist);
                        float intensity = costheta * attenuation;
                        intensity = GM_min(intensity, 1.0f);
                        SampleIntensityAccumulator += intensity;
                    }
                }
            }
        }

        all_light_direct[i] = SampleIntensityAccumulator * 0.25f;
        // Copy direct light values into global to prep for first light bounce
        all_light_global[i] = all_light_direct[i];
    }
}


// void delete_PostProcess_NeighbourSampling()
// {
//     // I can weight the samples by distance from the invalid one, non-linearly
//     for (int fy = 0; fy < FaceLightmap.h; ++fy)
//     {
//         for (int fx = 0; fx < FaceLightmap.w; ++fx)
//         {
//             int i = fy * FaceLightmap.w + fx;
//             float IndirLight = *(FaceLightmap.light_indirect + i);
//             if (IndirLight == LM_MARK_BAD_TEXEL)
//             {
//                 // try to sample from surrounding valid texels
//                 int NumValidNeighbourSamples = 0;
//                 float NeighbourSamplesAccumulator = 0.f;
//                 for (int y = fy - 1; y <= fy + 1; ++y)
//                 {
//                     if (y < 0 || y >= FaceLightmap.h)
//                         continue;

//                     for (int x = fx - 1; x <= fx + 1; ++x)
//                     {
//                         if (x < 0 || x >= FaceLightmap.w)
//                             continue;

//                         int j = y * FaceLightmap.w + x;
//                         float NeighbourIndir = *(FaceLightmap.light_indirect + j);
//                         if (NeighbourIndir != LM_MARK_BAD_TEXEL && NeighbourIndir > 0.f)
//                         {
//                             // sample the direct too
//                             float NeighbourDir = *(FaceLightmap.light_direct + j);
//                             NeighbourSamplesAccumulator += NeighbourDir + NeighbourIndir;
//                             ++NumValidNeighbourSamples;
//                         }
//                     }
//                 }

//                 if (NumValidNeighbourSamples > 0)
//                 {
//                     float AverageCombined = NeighbourSamplesAccumulator / (float)NumValidNeighbourSamples;
//                     // store the delta so that when we combine later we result with AverageCombined again
//                     *(FaceLightmap.light_indirect + i) = AverageCombined - *(FaceLightmap.light_direct + i);
//                 }
//                 else
//                 {
//                     *(FaceLightmap.light_indirect + i) = 0.f;
//                 }
//             }
//         }
//     }
// }


void lc_volume_baker_t::BakeLightCubes(game_map_build_data_t& BuildData)
{
    BuildDataShared = &BuildData;

    PlaceLightCubes();

    u32 NumSamples = LightCubeVolume.AmbientCubes.lenu() * 6; // num cubes * 6
    fixed_array<vec3> CubePositionsRepeated = fixed_array<vec3>(NumSamples, MemoryType::DefaultMalloc);
    fixed_array<vec3> CubeNormalsRepeated = fixed_array<vec3>(NumSamples, MemoryType::DefaultMalloc);
    CubePositionsRepeated.setlen(NumSamples);
    CubeNormalsRepeated.setlen(NumSamples);
    for (size_t i = 0; i < LightCubeVolume.AmbientCubes.lenu(); ++i)
    {
        vec3 CubePos = LightCubeVolume.CubePositions[i];
        CubePositionsRepeated[i*6+0] = CubePos;
        CubeNormalsRepeated[  i*6+0] = GM_FORWARD_VECTOR;
        CubePositionsRepeated[i*6+1] = CubePos;
        CubeNormalsRepeated[  i*6+1] = -GM_FORWARD_VECTOR;
        CubePositionsRepeated[i*6+2] = CubePos;
        CubeNormalsRepeated[  i*6+2] = GM_UP_VECTOR;
        CubePositionsRepeated[i*6+3] = CubePos;
        CubeNormalsRepeated[  i*6+3] = -GM_UP_VECTOR;
        CubePositionsRepeated[i*6+4] = CubePos;
        CubeNormalsRepeated[  i*6+4] = GM_RIGHT_VECTOR;
        CubePositionsRepeated[i*6+5] = CubePos;
        CubeNormalsRepeated[  i*6+5] = -GM_RIGHT_VECTOR;
    }

    RadiantBakeInfo.OutputLightmap = (float*)LightCubeVolume.AmbientCubes.data;
    RadiantBakeInfo.OutputLightmapSize = NumSamples;
    RadiantBakeInfo.LightMapTexelPositions = (radiant_vec3_t *)CubePositionsRepeated.data;
    RadiantBakeInfo.LightMapTexelNormals = (radiant_vec3_t *)CubeNormalsRepeated.data;
    RadiantBakeInfo.NumberOfSampleRaysPerTexel = 1024;
    RadiantBakeInfo.NumberOfLightBounces = 3;
    RadiantBakeInfo.BakeDirectLighting = false;

    RadiantBakeInfo.CacheDirectLightIndices = true;
    RadiantBakeInfo.OutputDirectLightIndices = (short *) LightCubeVolume.SignificantLightIndices.data;
    RadiantBakeInfo.OutputDirectLightIndicesSize = 4 * LightCubeVolume.SignificantLightIndices.lenu();
    RadiantBakeInfo.OutputDirectLightIndicesPerSample = 4;
    RadiantBakeInfo.DirectLightCachePositions = (radiant_vec3_t *)LightCubeVolume.CubePositions.data;
    RadiantBake(RadiantBakeInfo);

    LogMessage("- Baking light cubes finished");
    size_t TotalSizeLightVolumeData 
        = sizeof(vec3) * LightCubeVolume.CubePositions.lenu()
        + sizeof(lc_ambient_t) * LightCubeVolume.AmbientCubes.lenu()
        + sizeof(lc_light_indices_t) * LightCubeVolume.SignificantLightIndices.lenu();
    size_t TotalMallocedBytes = TotalSizeLightVolumeData 
        + CubePositionsRepeated.lenu() * sizeof(vec3)
        + CubeNormalsRepeated.lenu() * sizeof(vec3);
    LogMessage("      Generated %d cubes. Persistent size: %zd KB. Temporarily allocated %zd KB on heap.",
        LightCubeVolume.CubePositions.lenu(), TotalSizeLightVolumeData/1000, TotalMallocedBytes/1000);

    CubePositionsRepeated.free();
    CubeNormalsRepeated.free();
    arrfree(RadiantBakeInfo.WorldGeometryVertices);
    arrfree(RadiantBakeInfo.PointLights);

    BuildDataShared = nullptr;
}

size_t lc_volume_t::IndexByPosition(vec3 WorldPosition)
{
    vec3 OffsetFromStart = WorldPosition - Start;
    // translate WorldPosition so we go to nearest cube
    ivec3 Translate = LightCubePlacementInterval/2;
    ivec3 TranslatedWorldPos = ivec3((int)OffsetFromStart.x,(int)OffsetFromStart.y,(int)OffsetFromStart.z) + Translate;
    ivec3 XYZIndices;
    XYZIndices.x = TranslatedWorldPos.x / LightCubePlacementInterval.x;
    XYZIndices.y = TranslatedWorldPos.y / LightCubePlacementInterval.y;
    XYZIndices.z = TranslatedWorldPos.z / LightCubePlacementInterval.z;
    XYZIndices.x = std::max(0, XYZIndices.x);
    XYZIndices.y = std::max(0, XYZIndices.y);
    XYZIndices.z = std::max(0, XYZIndices.z);
    XYZIndices.x = std::min(XYZIndices.x, CountX-1);
    XYZIndices.y = std::min(XYZIndices.y, CountY-1);
    XYZIndices.z = std::min(XYZIndices.z, CountZ-1);
    const int VolumeAreaInCubes = CountX * CountZ;
    const int VolumeLengthInCubes = CountX;
    size_t CubeIndex = XYZIndices.y * VolumeAreaInCubes + XYZIndices.z * VolumeLengthInCubes + XYZIndices.x;
    ASSERT(0 <= CubeIndex && CubeIndex < CubePositions.lenu());
    ASSERT(CubePositions.lenu() == AmbientCubes.lenu());
    ASSERT(CubePositions.lenu() == SignificantLightIndices.lenu());
    return CubeIndex;
}

void lc_volume_t::Serialize(ByteBuffer *Buf)
{
    ByteBufferWrite(Buf, vec3, Start);
    ByteBufferWrite(Buf, vec3, End);
    ByteBufferWrite(Buf, int, CountX);
    ByteBufferWrite(Buf, int, CountY);
    ByteBufferWrite(Buf, int, CountZ);
    ByteBufferWrite(Buf, ivec3, LightCubePlacementInterval);

    u32 TotalCubeCount = u32(CountX * CountY * CountZ);
    ASSERT(TotalCubeCount == CubePositions.lenu());
    ASSERT(TotalCubeCount == AmbientCubes.lenu());
    ASSERT(TotalCubeCount == SignificantLightIndices.lenu());
    ByteBufferWrite(Buf, u32, TotalCubeCount);

    ByteBufferWriteBulk(Buf, CubePositions.data, sizeof(vec3) * TotalCubeCount);
    ByteBufferWriteBulk(Buf, AmbientCubes.data, sizeof(lc_ambient_t) * TotalCubeCount);
    ByteBufferWriteBulk(Buf, SignificantLightIndices.data, sizeof(lc_light_indices_t) * TotalCubeCount);

    ByteBufferWrite(Buf, u64, lc_volume_t_serialize_end_marker);
}

void lc_volume_t::Deserialize(ByteBuffer *Buf, MemoryType VolumeStorageType)
{
    ByteBufferRead(Buf, vec3, &Start);
    ByteBufferRead(Buf, vec3, &End);
    ByteBufferRead(Buf, int, &CountX);
    ByteBufferRead(Buf, int, &CountY);
    ByteBufferRead(Buf, int, &CountZ);
    ByteBufferRead(Buf, ivec3, &LightCubePlacementInterval);

    u32 TotalCubeCount;
    ByteBufferRead(Buf, u32, &TotalCubeCount);
    ASSERT(TotalCubeCount == u32(CountX * CountY * CountZ));

    CubePositions = fixed_array<vec3>(TotalCubeCount, VolumeStorageType);
    AmbientCubes = fixed_array<lc_ambient_t>(TotalCubeCount, VolumeStorageType);
    SignificantLightIndices = fixed_array<lc_light_indices_t>(TotalCubeCount, VolumeStorageType);
    CubePositions.setlen(TotalCubeCount);
    AmbientCubes.setlen(TotalCubeCount);
    SignificantLightIndices.setlen(TotalCubeCount);
    ByteBufferReadBulk(Buf, CubePositions.data, sizeof(vec3) * TotalCubeCount);
    ByteBufferReadBulk(Buf, AmbientCubes.data, sizeof(lc_ambient_t) * TotalCubeCount);
    ByteBufferReadBulk(Buf, SignificantLightIndices.data, sizeof(lc_light_indices_t) * TotalCubeCount);

    u64 SerializeEndMarker;
    ByteBufferRead(Buf, u64, &SerializeEndMarker);
    ASSERT(SerializeEndMarker == lc_volume_t_serialize_end_marker);
}

void lc_volume_baker_t::PlaceLightCubes()
{
    vec3 BoundsMinf = vec3();
    vec3 BoundsMaxf = vec3();

    for (int i = 0; i < BuildDataShared->TotalFaceCount; ++i)
    {
        MapEdit::Face *face = MapEdit::LevelEditorFaces[i];
        std::vector<MapEdit::Loop*> loopcycle = face->GetLoopCycle();
        for (MapEdit::Loop *loop : loopcycle)
        {
            const vec3 &p = loop->v->pos;
            BoundsMinf.x = fmin(p.x, BoundsMinf.x);
            BoundsMinf.y = fmin(p.y, BoundsMinf.y);
            BoundsMinf.z = fmin(p.z, BoundsMinf.z);
            BoundsMaxf.x = fmax(p.x, BoundsMaxf.x);
            BoundsMaxf.y = fmax(p.y, BoundsMaxf.y);
            BoundsMaxf.z = fmax(p.z, BoundsMaxf.z);
        }
    }

    ivec3 Start;
    ivec3 End;
    Start.x = GM_sign(BoundsMinf.x) * (int(ceilf(fabsf(BoundsMinf.x))) - int(ceilf(fabsf(BoundsMinf.x))) % LightCubeVolume.LightCubePlacementInterval.x);
    Start.y = GM_sign(BoundsMinf.y) * (int(ceilf(fabsf(BoundsMinf.y))) - int(ceilf(fabsf(BoundsMinf.y))) % LightCubeVolume.LightCubePlacementInterval.y);
    Start.z = GM_sign(BoundsMinf.z) * (int(ceilf(fabsf(BoundsMinf.z))) - int(ceilf(fabsf(BoundsMinf.z))) % LightCubeVolume.LightCubePlacementInterval.z);
    End.x = GM_sign(BoundsMaxf.x) * (int(ceilf(fabsf(BoundsMaxf.x))) - int(ceilf(fabsf(BoundsMaxf.x))) % LightCubeVolume.LightCubePlacementInterval.x);
    End.y = GM_sign(BoundsMaxf.y) * (int(ceilf(fabsf(BoundsMaxf.y))) - int(ceilf(fabsf(BoundsMaxf.y))) % LightCubeVolume.LightCubePlacementInterval.y);
    End.z = GM_sign(BoundsMaxf.z) * (int(ceilf(fabsf(BoundsMaxf.z))) - int(ceilf(fabsf(BoundsMaxf.z))) % LightCubeVolume.LightCubePlacementInterval.z);

    LightCubeVolume.CountX = (End.x/LightCubeVolume.LightCubePlacementInterval.x - Start.x/LightCubeVolume.LightCubePlacementInterval.x + 1);
    LightCubeVolume.CountY = (End.y/LightCubeVolume.LightCubePlacementInterval.y - Start.y/LightCubeVolume.LightCubePlacementInterval.y + 1);
    LightCubeVolume.CountZ = (End.z/LightCubeVolume.LightCubePlacementInterval.z - Start.z/LightCubeVolume.LightCubePlacementInterval.z + 1);
    LightCubeVolume.Start = vec3((float)Start.x,(float)Start.y,(float)Start.z);
    LightCubeVolume.End = vec3((float)End.x,(float)End.y,(float)End.z);
    LightCubeVolume.Start += CubePlacementOffsetToAvoidClipping;
    LightCubeVolume.End += CubePlacementOffsetToAvoidClipping;
    int TotalNumberOfCubes = LightCubeVolume.CountX * LightCubeVolume.CountY * LightCubeVolume.CountZ;
    ASSERT(TotalNumberOfCubes >= 0);
    ASSERT(TotalNumberOfCubes <= 1000000);

    LightCubeVolume.CubePositions = fixed_array<vec3>(TotalNumberOfCubes, MemoryType::DefaultMalloc);
    LightCubeVolume.AmbientCubes = fixed_array<lc_ambient_t>(TotalNumberOfCubes, MemoryType::DefaultMalloc);
    LightCubeVolume.SignificantLightIndices = fixed_array<lc_light_indices_t>(TotalNumberOfCubes, MemoryType::DefaultMalloc);
    LightCubeVolume.CubePositions.setlen(TotalNumberOfCubes);
    LightCubeVolume.AmbientCubes.setlen(TotalNumberOfCubes);
    LightCubeVolume.SignificantLightIndices.setlen(TotalNumberOfCubes);

    const float ErrorTolerance = 0.1f;
    for (float y = LightCubeVolume.Start.y; y <= LightCubeVolume.End.y + ErrorTolerance; y += (float)LightCubeVolume.LightCubePlacementInterval.y)
    {
        for (float z = LightCubeVolume.Start.z; z <= LightCubeVolume.End.z + ErrorTolerance; z += (float)LightCubeVolume.LightCubePlacementInterval.z)
        {
            for (float x = LightCubeVolume.Start.x; x <= LightCubeVolume.End.x + ErrorTolerance; x += (float)LightCubeVolume.LightCubePlacementInterval.x)
            {
                size_t CubeIndex = LightCubeVolume.IndexByPosition(vec3(x,y,z));
                LightCubeVolume.CubePositions[CubeIndex] = (vec3(x,y,z));
                LightCubeVolume.AmbientCubes[CubeIndex] = (lc_ambient_t());
                LightCubeVolume.SignificantLightIndices[CubeIndex] = (lc_light_indices_t());
            }
        }
    }

    ASSERT(LightCubeVolume.CubePositions.lenu() == LightCubeVolume.CubePositions.cap());
    ASSERT(LightCubeVolume.AmbientCubes.lenu() == LightCubeVolume.AmbientCubes.cap());
    ASSERT(LightCubeVolume.SignificantLightIndices.lenu() == LightCubeVolume.SignificantLightIndices.cap());
}

