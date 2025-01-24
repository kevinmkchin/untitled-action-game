

vec3 *all_lm_pos = NULL;
vec3 *all_lm_norm = NULL;
vec3 *all_lm_tangent = NULL;
// vec3 *all_patches_id = NULL;
float *all_light_global = NULL;
float *all_light_direct = NULL;
float *all_light_indirect = NULL;



vec3 TestLightSource = vec3(0, 110, 0);
vec3 TestLightSource2 = vec3(150, 110, 0);
LevelPolygonOctree LightMapOcclusionTree;
const float LightMapTexelSize = 8.f; // in world units
const int HemicubeFaceW = 100;
const int HemicubeFaceH = HemicubeFaceW;
const int HemicubeFaceWHalf = HemicubeFaceW/2;
const int HemicubeFaceHHalf = HemicubeFaceH/2;


void ThreadSafe_DoDirectLightingIntoLightMap(u32 patchIndexStart, u32 patchIndexEnd)
{
    // calculate direct lighting for patchIndexStart upto and including patchIndexEnd-1

    // 4x multisampling with sparse regular grid distribution
    //  ____ 
    // | *  |
    // |   *|
    // |*   |
    // |  * |
    //  ----

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
#if SUNLIGHT_TEST
            vec3 incidence_ray = vec3(-1.0f, 0.9f, -0.16f); // direction to sun
            // vec3 incidence_ray = vec3(-0.16f, 0.9f, -0.16f); // direction to sun
            float costheta = Dot(Normalize(incidence_ray), patch_normal);
            if (costheta > 0.f)
            {
                // occlusion test
                LineCollider ray_collider;
                ray_collider.a = patch_position + Normalize(incidence_ray) * 32000.0f;
                ray_collider.b = patch_position + Normalize(incidence_ray) * 0.2f;
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
#else
            vec3 incidence_ray = TestLightSource - SamplePositions[sample];
            float costheta = Dot(Normalize(incidence_ray), patch_normal);
            if (costheta > 0.f)
            {
                float dist = Magnitude(incidence_ray);

                // occlusion test
                LineCollider ray_collider;
                ray_collider.a = TestLightSource;
                ray_collider.b = TestLightSource - incidence_ray * 0.98f;
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

            // vec3 incidence_ray2 = TestLightSource2 - SamplePositions[sample];
            // costheta = Dot(Normalize(incidence_ray2), patch_normal);
            // if (costheta > 0.f)
            // {
            //     float dist = Magnitude(incidence_ray2);

            //     // occlusion test
            //     LineCollider ray_collider;
            //     ray_collider.a = TestLightSource2;
            //     ray_collider.b = TestLightSource2 - incidence_ray2 * 0.98f;
            //     bool occluded = false;

            //     if (LightMapOcclusionTree.Query(ray_collider))
            //     {
            //         occluded = true;
            //     }

            //     if (!occluded)
            //     {
            //         float atten_lin = 0.022f;
            //         float atten_quad = 0.0019f;
            //         float attenuation = 1.f / 
            //             (1.f + atten_lin * dist + atten_quad * dist * dist);
            //         float intensity = costheta * attenuation;
            //         intensity = GM_min(intensity, 1.0f);
            //         SampleIntensityAccumulator += intensity;
            //     }
            // }
#endif
        }

        all_light_direct[i] = SampleIntensityAccumulator * 0.25f;
        // Copy direct light values into global to prep for first light bounce
        all_light_global[i] = all_light_direct[i];
    }
}


void BakeStaticLighting(game_map_build_data_t& BuildData)
{
    const int totalfacecount = BuildData.TotalFaceCount;
    std::unordered_map<u32, std::vector<float>>& VertexBuffers = BuildData.VertexBuffers;
    std::vector<vec3>& ColliderWorldPoints = BuildData.ColliderWorldPoints;
    std::vector<u32>& ColliderSpans = BuildData.ColliderSpans;

    // === LIGHT MAP BAKING ===
    // As level gets bigger, I need to find a way to reduce patch count. Otherwise
    // time to bake will continue increasing linearly.
    Bounds MapBounds = Bounds(vec3(-0.17f, -0.17f, -0.17f), vec3(8000, 8000, 8000));
    LightMapOcclusionTree = LevelPolygonOctree(MapBounds, 100, 24);
    std::vector<FlatPolygonCollider> MapSurfaceColliders(ColliderSpans.size());
    int iter = 0;
    // later, when only some surfaces have colliders, can't use ColliderSpan, need to traverse all faces again
    for (u32 i = 0; i < ColliderSpans.size(); ++i)
    {
        u32 span = ColliderSpans[i];
        FlatPolygonCollider& surface = MapSurfaceColliders[i];
        surface.pointCloudPtr = &ColliderWorldPoints[iter];
        surface.pointCount = span;
        surface.debugId = i;
        iter += span;
        LightMapOcclusionTree.Insert(&surface);
    }
    // LightMapOcclusionTree is complete

    // Single bounce radiosity
    // Start with emitting patches, of which there are N.
    // Each emitting patch goes every other patch, of which there are M. O(NM)
    // Then, each patch (M) goes to every other patch (M) and transfers energy exactly once. O(NMM) 

    stbrp_rect *lm_rects = NULL;
    // calculate bounds, and divide into patches
    arrsetcap(all_lm_pos, 625000); // patch num cap should be 256^3-1 so about 16 million patches
    arrsetcap(all_lm_norm, 625000);
    arrsetcap(all_light_global, 625000);
    arrsetcap(all_lm_tangent, 625000);
    // arrsetcap(all_patches_id, 625000);
    arrsetcap(all_light_direct, 625000);
    arrsetcap(all_light_indirect, 625000);
    // u32 patchCounter = 0;
    for (int i = 0; i < totalfacecount; ++i)
    {
        MapEdit::Face *face = MapEdit::LevelEditorFaces.At(i);
        
        lm_face_t lm;
        // project 3D verts unto 2D plane using basis vectors U and V
        // find min uv and max uv
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
        // Resizing the light map dimensions to be 2 units wider and taller (by moving minuv and maxuv)
        // is the best way to add padding for bilinear filtering. No padding means filtering samples light
        // maps not belonging to this surface (since everything gets packed into an atlas) and having the
        // padding copy the corresponding adjacent light map value will create unnatural seams where
        // surfaces are supposed to connect. By simply making the light maps larger than needed and having
        // these virtual patches be sampled too creates the most natural bilinear filtering results.
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
        }

        // lm data
        lm.w = (i32)(dim.x / LightMapTexelSize);
        lm.h = (i32)(dim.y / LightMapTexelSize);
        i32 lmsz = lm.w*lm.h;
        lm.pos = arraddnptr(all_lm_pos, lmsz);
        lm.norm = arraddnptr(all_lm_norm, lmsz);
        lm.light = arraddnptr(all_light_global, lmsz);
        lm.tangent = arraddnptr(all_lm_tangent, lmsz);
        // lm.patches_id = arraddnptr(all_patches_id, lmsz);
        float *lm_direct = arraddnptr(all_light_direct, lmsz);
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
            // lm.patches_id[pi] = HandleIdToRGB(++patchCounter);
            lm.light[pi] = 0.f;
            lm_direct[pi] = 0.f;
            lm.light_indirect[pi] = 0.f;
        }

        face->lightmap = lm;
        stbrp_rect rect;
        rect.id = face->storageIndex;
        rect.w = lm.w;
        rect.h = lm.h;
        arrput(lm_rects, rect);
    }

    // Pack light maps
    i32 lightMapAtlasW = 4096;
    i32 lightMapAtlasH = 4096;
    stbrp_node *LMPackerNodes = NULL;
    arrsetlen(LMPackerNodes, lightMapAtlasW);
    stbrp_context LightMapPacker;
    stbrp_init_target(&LightMapPacker, lightMapAtlasW, lightMapAtlasH, LMPackerNodes, (int)arrlenu(LMPackerNodes));
    stbrp_pack_rects(&LightMapPacker, lm_rects, (int)arrlenu(lm_rects));
    arrfree(LMPackerNodes);
    size_t LightMapRectsCount = arrlenu(lm_rects);
    for (size_t i = 0; i < LightMapRectsCount; ++i)
    {
        stbrp_rect rect = lm_rects[i];
        if (rect.was_packed == 0) continue;
        ASSERT(rect.was_packed != 0); // TODO(Kevin): additional light map atlases if couldn't fit into one

        vec2 minuv = vec2((float)(rect.x) / (float)lightMapAtlasW, (float)(rect.y) / (float)lightMapAtlasH);
        vec2 maxuv = vec2((float)(rect.x + rect.w) / (float)lightMapAtlasW, (float)(rect.y + rect.h) / (float)lightMapAtlasH);

        int editorFaceIndex = rect.id;
        MapEdit::Face *face = MapEdit::LevelEditorFaces.At(editorFaceIndex);
        std::vector<MapEdit::Loop*> loopcycle = face->GetLoopCycle();
        for (MapEdit::Loop *loop : loopcycle)
        {
            // map lm uv from local to global in light map atlas
            loop->lmuvcache.x = Lerp(minuv.x, maxuv.x, loop->lmuvcache.x); 
            loop->lmuvcache.y = Lerp(minuv.y, maxuv.y, loop->lmuvcache.y); 
        }
    }

    // Sort faces by their textures and generate vertex buffers
    for (int i = 0; i < totalfacecount; ++i)
    {
        MapEdit::Face *face = MapEdit::LevelEditorFaces.At(i);
        db_tex_t tex = face->texture;
        if (VertexBuffers.find(tex.persistId) == VertexBuffers.end())
        {
            VertexBuffers.emplace(tex.persistId, std::vector<float>());
        }

        std::vector<float>& vb = VertexBuffers.at(tex.persistId);
        TriangulateFace_ForFaceBatch_QuickDumb(*face, &vb);
    }


    // direct lighting from emissive points and patches
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

    std::thread t0 = std::thread(ThreadSafe_DoDirectLightingIntoLightMap, 0, progress10);
    std::thread t1 = std::thread(ThreadSafe_DoDirectLightingIntoLightMap, progress10, progress20);
    std::thread t2 = std::thread(ThreadSafe_DoDirectLightingIntoLightMap, progress20, progress30);
    std::thread t3 = std::thread(ThreadSafe_DoDirectLightingIntoLightMap, progress30, progress40);
    std::thread t4 = std::thread(ThreadSafe_DoDirectLightingIntoLightMap, progress40, progress50);
    std::thread t5 = std::thread(ThreadSafe_DoDirectLightingIntoLightMap, progress50, progress60);
    std::thread t6 = std::thread(ThreadSafe_DoDirectLightingIntoLightMap, progress60, progress70);
    std::thread t7 = std::thread(ThreadSafe_DoDirectLightingIntoLightMap, progress70, progress80);
    std::thread t8 = std::thread(ThreadSafe_DoDirectLightingIntoLightMap, progress80, progress90);
    std::thread t9 = std::thread(ThreadSafe_DoDirectLightingIntoLightMap, progress90, numpatches);
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

    // indirect lighting

    face_batch_t SceneLightingModel;

    float *patches_vb = NULL; 
    for (auto& vbpair : VertexBuffers)
    {
        const std::vector<float>& vb = vbpair.second;
        float *writeto = arraddnptr(patches_vb, vb.size());
        memcpy(writeto, vb.data(), sizeof(float)*vb.size());
    }
    CreateFaceBatch(&SceneLightingModel);
    RebindFaceBatch(&SceneLightingModel, u32(sizeof(float)*arrlenu(patches_vb)), patches_vb);
    arrfree(patches_vb);
    SceneLightingModel.ColorTexture = Assets.DefaultMissingTexture;

    // DECLARE LIGHT MAP ATLAS
    float *LIGHT_MAP_ATLAS = NULL;
    arrsetcap(LIGHT_MAP_ATLAS, lightMapAtlasW*lightMapAtlasH);
    CreateGPUTextureFromBitmap(&SceneLightingModel.LightMapTexture, (void*)LIGHT_MAP_ATLAS, 
        lightMapAtlasW, lightMapAtlasH, GL_R32F, GL_RED, GL_LINEAR, GL_LINEAR, GL_FLOAT);


    const int HemicubeFaceArea = HemicubeFaceW * HemicubeFaceH;
    const int HemicubeFaceAreaHalf = HemicubeFaceW * HemicubeFaceHHalf;
    const float PixelAreaInSolidAngle = GM_PI / float(2 * HemicubeFaceW * HemicubeFaceH);
    GPUFrameBuffer HemicubeFBO; // the cube faces are laid out horizontally
    HemicubeFBO.width = HemicubeFaceW*5;
    HemicubeFBO.height = HemicubeFaceH;
    CreateGPUFrameBuffer(&HemicubeFBO, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    u32 HemicubePBO;
    glGenBuffers(1, &HemicubePBO);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, HemicubePBO);
    const GLsizeiptr PBOSz = HemicubeFaceW * HemicubeFaceH * 4;
    glBufferData(GL_PIXEL_PACK_BUFFER, 5*PBOSz*sizeof(float), NULL, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    UseShader(PatchesIDShader);

    float HEMICUBE_NEARCLIP = 1.f;
    mat4 HemicubePerspectiveMatrix = ProjectionMatrixPerspective(90.f*GM_DEG2RAD, 1.f, HEMICUBE_NEARCLIP, GAMEPROJECTION_FARCLIP);
    GLBindMatrix4fv(PatchesIDShader, "projMatrix", 1, HemicubePerspectiveMatrix.ptr());

    glBindFramebuffer(GL_FRAMEBUFFER, HemicubeFBO.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0); // Asynchronously read pixel data into the PBO
    glBindBuffer(GL_PIXEL_PACK_BUFFER, HemicubePBO);

    static float MultiplierMapTop[HemicubeFaceW*HemicubeFaceH];
    static float MultiplierMapSide[HemicubeFaceW*HemicubeFaceHHalf];
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


    for (int bounces = 0; bounces < 2; ++bounces)
    {
        // Copy the radiance info thus far to use for first light bounce
        // Would be just direct lighting for first bounce, but future bounces
        // would be direct light + accumulated indirect light.
        for (size_t i = 0; i < LightMapRectsCount; ++i)
        {
            stbrp_rect rect = lm_rects[i];
            if (rect.was_packed == 0) continue;
            ASSERT(rect.was_packed != 0);
            MapEdit::Face *face = MapEdit::LevelEditorFaces.At(rect.id);
            lm_face_t lmface = face->lightmap;
            BlitRect((u8*)LIGHT_MAP_ATLAS, lightMapAtlasW, lightMapAtlasH, 
                (u8*)lmface.light, lmface.w, lmface.h, rect.x, rect.y, sizeof(float));
        }
        // Update SceneLightingModel to use the updated radiance information
        UpdateGPUTextureFromBitmap(&SceneLightingModel.LightMapTexture, (u8*)LIGHT_MAP_ATLAS,
            lightMapAtlasW, lightMapAtlasH);


        for (int FaceIndex = 0; FaceIndex < MapEdit::LevelEditorFaces.count; ++FaceIndex)
        {
            MapEdit::Face *Face = MapEdit::LevelEditorFaces.At(FaceIndex);
            lm_face_t& FaceLightMap = Face->lightmap;

            // TODO reset "cache" - just an array for now

            u32 NumTexelsOnFace = FaceLightMap.w * FaceLightMap.h;
            for (u32 FaceTexelIndex = 0; FaceTexelIndex < NumTexelsOnFace; ++FaceTexelIndex)
            {
                // if (FaceIndex == 1 && FaceTexelIndex >= 50 && FaceTexelIndex < 52)
                // {
                //     if (RDOCAPI) RDOCAPI->StartFrameCapture(NULL, NULL);
                // }
                
                vec3 patch_i_pos = *(FaceLightMap.pos + FaceTexelIndex); // rename to texel
                vec3 patch_i_normal = *(FaceLightMap.norm + FaceTexelIndex);
                vec3 patch_i_basisV = *(FaceLightMap.tangent + FaceTexelIndex);

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

                if (RDOCAPI) RDOCAPI->EndFrameCapture(NULL, NULL);

                // Putting the glReadPixels together at the end of the draw calls is somehow appreciably faster
                glReadPixels(0, 0, HemicubeFaceW, HemicubeFaceH, GL_RGBA, GL_FLOAT, 0);
                glReadPixels(HemicubeFaceW*1, 0, HemicubeFaceW, HemicubeFaceH, GL_RGBA, GL_FLOAT, (void*)(PBOSz*sizeof(float)*1));
                glReadPixels(HemicubeFaceW*2, 0, HemicubeFaceW, HemicubeFaceH, GL_RGBA, GL_FLOAT, (void*)(PBOSz*sizeof(float)*2));
                glReadPixels(HemicubeFaceW*3, 0, HemicubeFaceW, HemicubeFaceH, GL_RGBA, GL_FLOAT, (void*)(PBOSz*sizeof(float)*3));
                glReadPixels(HemicubeFaceW*4, 0, HemicubeFaceW, HemicubeFaceH, GL_RGBA, GL_FLOAT, (void*)(PBOSz*sizeof(float)*4));


                // Map the PBO to access pixel data in system memory
                float *FrontFaceData = (float*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY); 
                ASSERT(FrontFaceData);
                float *UpFaceData = FrontFaceData + PBOSz;
                float *DownFaceData = FrontFaceData + PBOSz*2;
                float *LeftFaceData = FrontFaceData + PBOSz*3;
                float *RightFaceData = FrontFaceData + PBOSz*4;


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

                    if (HemicubePixel.w == 0.69f)
                        ++BackfacePixelCount;

                    // Side faces
                    if (p < HemicubeFaceAreaHalf)
                        continue;

                    HemicubePixel = { UpFaceData[p*4], UpFaceData[p*4+1], 
                        UpFaceData[p*4+2], UpFaceData[p*4+3] };
                    Radiance = HemicubePixel.x;
                    DifferentialFormFactor = MultiplierMapSide[p-HemicubeFaceAreaHalf];
                    RadiositiesAccumulator += DifferentialFormFactor * Radiance;

                    if (HemicubePixel.w == 0.69f)
                        ++BackfacePixelCount;

                    HemicubePixel = { DownFaceData[p*4], DownFaceData[p*4+1], 
                        DownFaceData[p*4+2], DownFaceData[p*4+3] };
                    Radiance = HemicubePixel.x;
                    DifferentialFormFactor = MultiplierMapSide[p-HemicubeFaceAreaHalf];
                    RadiositiesAccumulator += DifferentialFormFactor * Radiance;

                    if (HemicubePixel.w == 0.69f)
                        ++BackfacePixelCount;

                    HemicubePixel = { LeftFaceData[p*4], LeftFaceData[p*4+1], 
                        LeftFaceData[p*4+2], LeftFaceData[p*4+3] };
                    Radiance = HemicubePixel.x;
                    DifferentialFormFactor = MultiplierMapSide[p-HemicubeFaceAreaHalf];
                    RadiositiesAccumulator += DifferentialFormFactor * Radiance;

                    if (HemicubePixel.w == 0.69f)
                        ++BackfacePixelCount;

                    HemicubePixel = { RightFaceData[p*4], RightFaceData[p*4+1], 
                        RightFaceData[p*4+2], RightFaceData[p*4+3] };
                    Radiance = HemicubePixel.x;
                    DifferentialFormFactor = MultiplierMapSide[p-HemicubeFaceAreaHalf];
                    RadiositiesAccumulator += DifferentialFormFactor * Radiance;
                    
                    if (HemicubePixel.w == 0.69f)
                        ++BackfacePixelCount;

                    if (BackfacePixelCount > BackfaceTolerance)
                        break;
                }

                if (BackfacePixelCount > BackfaceTolerance)
                {
                    // TODO then what do I do about this texel?
                    RadiositiesAccumulator = 0.f;
                }

                *(FaceLightMap.light_indirect + FaceTexelIndex) = RadiositiesAccumulator;

                glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            }

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
            //     FaceLightMap.light_indirect[i] = IrradianceAtTexel;
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

    // Final blit to light map atlas
    for (size_t i = 0; i < LightMapRectsCount; ++i)
    {
        stbrp_rect rect = lm_rects[i];
        if (rect.was_packed == 0) continue;
        ASSERT(rect.was_packed != 0);
        MapEdit::Face *face = MapEdit::LevelEditorFaces.At(rect.id);
        lm_face_t lmface = face->lightmap;
        BlitRect((u8*)LIGHT_MAP_ATLAS, lightMapAtlasW, lightMapAtlasH, 
            (u8*)lmface.light, lmface.w, lmface.h, rect.x, rect.y, sizeof(float));
    }
    ByteBufferWrite(&BuildData.Output, i32, lightMapAtlasW);
    ByteBufferWrite(&BuildData.Output, i32, lightMapAtlasH);
    ByteBufferWriteBulk(&BuildData.Output, LIGHT_MAP_ATLAS, lightMapAtlasW*lightMapAtlasH*sizeof(float));
    arrfree(LIGHT_MAP_ATLAS);

    arrfree(lm_rects);
    arrfree(all_lm_pos);
    arrfree(all_lm_norm);
    arrfree(all_light_global);
    // arrfree(all_patches_id);
    arrfree(all_light_direct);
    arrfree(all_light_indirect);
}

