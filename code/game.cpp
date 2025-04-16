#include "game.h"
#include "renderer.h"
#include "physics.h"
#include "nav.h"
#include "gui.h"
#include "debugmenu.h"
#include "saveloadlevel.h"
#include "enemy.h"


#if INTERNAL_BUILD
static recast_debug_draw_gl3_t RecastDebugDrawer;
#endif // INTERNAL_BUILD
#ifdef JPH_DEBUG_RENDERER
#include "physics_debug.h"
static jph_debug_draw_gl3_t *JoltDebugDrawer;
#endif // JPH_DEBUG_RENDERER
static constexpr float GAMEPROJECTION_NEARCLIP = 4.f; // even 2 works fine to remove z fighting
static constexpr float GAMEPROJECTION_FARCLIP = 3200.f;

static game_state *GameState;


void InitializeGame(app_state *AppState)
{
    GameState = AppState->GameState;

    GameState->AnimatorPool = fixed_array<animator_t>(64, MemoryType::Game);
    GameState->AnimatorPool.setlen(64);
    for (size_t i = 0; i < GameState->AnimatorPool.length; ++i)
        GameState->AnimatorPool[i] = animator_t();

    Physics.Initialize();
    Physics.VirtualCharacterContactListener.GameState = GameState;

    SetupProjectilesDataAndAllocateMemory();
    GameState->BloodParticles.Alloc(128, 32, MemoryType::Game);
    persistent_vertex_stream::vertex_desc ParticleVertexDesc;
    ParticleVertexDesc.VByteSize = sizeof(particle_vertex);
    ParticleVertexDesc.VAttrib0_Format = GL_FLOAT;
    ParticleVertexDesc.VAttrib0_Size = 3;
    ParticleVertexDesc.VAttrib0_Offset = 0;
    ParticleVertexDesc.VAttrib1_Format = GL_FLOAT;
    ParticleVertexDesc.VAttrib1_Size = 4;
    ParticleVertexDesc.VAttrib1_Offset = offsetof(particle_vertex, Color);
    ParticleVertexDesc.VAttrib2_Format = GL_FLOAT;
    ParticleVertexDesc.VAttrib2_Size = 2;
    ParticleVertexDesc.VAttrib2_Offset = offsetof(particle_vertex, UV);
    GameState->BloodParticlesVB.Alloc(128*6, ParticleVertexDesc);
    GameState->PQuadBuf = fixed_array<particle_vertex>(256*6, MemoryType::Game);

    EnemySystem.Init();

    GameState->Player.Init();

#if INTERNAL_BUILD
    RecastDebugDrawer.GameState = GameState;
    RecastDebugDrawer.SupportRenderer = AppState->PrimitivesRenderer;
#endif // INTERNAL_BUILD
#ifdef JPH_DEBUG_RENDERER
    JoltDebugDrawer = new_InGameMemory(jph_debug_draw_gl3_t)();
    JoltDebugDrawer->SupportRenderer = AppState->PrimitivesRenderer;
#endif // JPH_DEBUG_RENDERER
}

void DestroyGame()
{
    EnemySystem.Destroy();
    Physics.Destroy();

    GameState->BloodParticlesVB.Free();
}

void LoadLevel(const char *MapPath)
{
    StaticLevelMemory.ArenaOffset = 0;
    GameState->StaticInstances = fixed_array<model_instance_data_t>(MaxStaticInstances, MemoryType::Level);

    if (GameState->LevelLoaded)
        UnloadPreviousLevel();

    if (LoadGameMap(GameState, MapPath) == false)
    {
        LogError("Failed to load game map: %s", MapPath);
        return;
    }

    ASSERT(Physics.PhysicsSystem);
    CreateAndRegisterLevelCollider(GameState);
    ASSERT(CreateRecastNavMesh(GameState));

    GameState->Player.CharacterController->SetPosition(ToJoltVector(GameState->PlayerStartPosition));
    // TODO Apply rotation to camera rotation instead
    // GameState->Player.mCharacter->SetRotation(ToJoltQuat(EulerToQuat(GameState->PlayerStartRotation)));

    GameState->LevelLoaded = true;
}

void UnloadPreviousLevel()
{
    if (!GameState->LevelColliderBodyId.IsInvalid())
    {
        Physics.BodyInterface->RemoveBody(GameState->LevelColliderBodyId);
        Physics.BodyInterface->DestroyBody(GameState->LevelColliderBodyId);
    }

    EnemySystem.RemoveAll();
    DestroyRecastNavMesh();

    GameState->LevelLoaded = false;
    StaticLevelMemory.ArenaOffset = 0;
}

void NonPhysicsTick()
{
    if (!(EnemySystem.Enemies[0].Flags & EnemyFlag_Active))
    {
        EnemySystem.SpawnEnemy(GameState);
        while (EnemySystem.Enemies[0].Position.y > 10.f)
        {
            GetRandomPointOnNavMesh((float*)&EnemySystem.Enemies[0].Position);
        }
    }

    NonPhysicsTickAllEnemies(GameState);

    NonPhysicsUpdateProjectiles(GameState);

    GameState->Player.HandleInput(GameState->AppState);

    if (GameState->Player.Health <= 0.f)
    {
        // ASSERT(0);
    }
    GUI::PrimitiveTextFmt(20, 180, 9, GUI::Align::LEFT, "kills %d", GameState->KillEnemyCounter);
    GUI::PrimitiveTextFmt(180, 650, 18, GUI::Align::RIGHT, "%d", (int)ceilf(GameState->Player.Health));

    if (DebugShowNumberOfPhysicsBodies)
    {
        GUI::PrimitiveTextFmt(8, 48, GUI::GetFontSize(), GUI::Align::LEFT, 
            "JPH NumActiveBodies: %d", Physics.PhysicsSystem->GetNumActiveBodies(JPH::EBodyType::RigidBody));
    }
}

void PrePhysicsTick()
{
    // particle_emitter FountainTest;
    // FountainTest.WorldP = vec3(0.f,0.f,0.f);
    // FountainTest.PSpread = vec3(0.f,0.f,0.f);
    // FountainTest.dP = vec3(0.f,220.f,0.f);
    // FountainTest.dPSpread = vec3(32.f,0.f,32.f);
    // FountainTest.ddP = vec3(0.f,FromJoltUnit(-9.8f),0.f);
    // FountainTest.Color = vec4(1,1,1,1.4f);
    // FountainTest.ColorSpread = vec4(0,0,0,0.1f);
    // FountainTest.dColor = vec4(0,0,0,-1.35f);
    // FountainTest.Timer = 0.f;
    // FountainTest.ParticleLifeTimer = 2.f;
    // GameState->BloodParticles.Emitters.put(FountainTest);

    PrePhysicsTickAllEnemies(GameState);

    PrePhysicsUpdateProjectiles(GameState);

    GameState->Player.PrePhysicsUpdate(GameState);
}

void PostPhysicsTick()
{
    PostPhysicsTickAllEnemies(GameState);

    PostPhysicsUpdateProjectiles(GameState);

    GameState->Player.PostPhysicsUpdate();
}

void LateNonPhysicsTick()
{
    GameState->Player.LateNonPhysicsTick();

    random_series &EmitterRNG = GameState->ParticlesRNG;
    UpdateParticles(GameState->BloodParticles, EmitterRNG);
}

void DoGameLoop(app_state *AppState)
{
    GameState->DynamicInstances = fixed_array<model_instance_data_t>(MaxDynamicInstances, MemoryType::Frame);

    if (GameState->GameLoopCanRun) 
    {
        NonPhysicsTick();

        static float Accumulator = 0.f;
        Accumulator += DeltaTime;
        while (Accumulator >= FixedDeltaTime)
        {
            PrePhysicsTick();
            Physics.Tick();
            PostPhysicsTick();

            Accumulator -= FixedDeltaTime;
        }

        LateNonPhysicsTick();

        for (size_t i = 0; i < GameState->AnimatorPool.length; ++i)
        {
            if (GameState->AnimatorPool[i].HasOwner)
            {
                GameState->AnimatorPool[i].UpdateGlobalPoses(DeltaTime);
                GameState->AnimatorPool[i].CalculateSkinningMatrixPalette();
            }
        }
    }

    RequestDrawGame();

    if (GameState->GameLoopCanRun) 
    {
        UpdateGameGUI(AppState->GUIRenderTargetWidth, AppState->GUIRenderTargetHeight);
    }
}

void UpdateGameGUI(i32 GUIWidth, i32 GUIHeight)
{
    // temp crosshair
    ivec2 guiwh = ivec2(GUIWidth, GUIHeight);
    GUI::PrimitivePanel(GUI::UIRect(guiwh.x / 2 - 3, guiwh.y / 2 - 3, 6, 6), vec4(0, 0, 0, 1));
    GUI::PrimitivePanel(GUI::UIRect(guiwh.x / 2 - 2, guiwh.y / 2 - 2, 4, 4), vec4(1, 1, 1, 1));

    if (DebugShowGameMemoryUsage)
    {
        GUI::PrimitiveTextFmt(8, 58, GUI::GetFontSize(), GUI::Align::LEFT, 
            "Game Memory usage  [%zd/%zd KB]", 
            StaticGameMemory.ArenaOffset/1000, StaticGameMemory.ArenaSize/1000);
        GUI::PrimitiveTextFmt(8, 68, GUI::GetFontSize(), GUI::Align::LEFT, 
            "Level Memory usage [%zd/%zd KB]", 
            StaticLevelMemory.ArenaOffset/1000, StaticLevelMemory.ArenaSize/1000);
        GUI::PrimitiveTextFmt(8, 78, GUI::GetFontSize(), GUI::Align::LEFT, 
            "Frame Memory usage [%zd/%zd KB]", 
            FrameMemory.ArenaOffset/1000, FrameMemory.ArenaSize/1000);
    }
}

void DebugDrawGame()
{
    //vec3 p = GameState->Player.Root;// +GameState->Player.CamOffsetFromRoot;
    // size_t pi = LightCacheVolume->IndexByPosition(p);
    // for (size_t i = 0; i < GameState->LightCacheVolume->CubePositions.lenu(); ++i)
    // {
    //     const vec3 &CubePos = GameState->LightCacheVolume->CubePositions[i];
    //     lc_ambient_t &AmbientCube = GameState->LightCacheVolume->AmbientCubes[i];
    //     SupportRenderer.DrawColoredCube(CubePos, 1.f, 
    //         vec4(AmbientCube.PosX,AmbientCube.PosX,AmbientCube.PosX,1),
    //         vec4(AmbientCube.NegX,AmbientCube.NegX,AmbientCube.NegX,1),
    //         vec4(AmbientCube.PosY,AmbientCube.PosY,AmbientCube.PosY,1),
    //         vec4(AmbientCube.NegY,AmbientCube.NegY,AmbientCube.NegY,1),
    //         vec4(AmbientCube.PosZ,AmbientCube.PosZ,AmbientCube.PosZ,1),
    //         vec4(AmbientCube.NegZ,AmbientCube.NegZ,AmbientCube.NegZ,1)
    //     );
    // }

#if INTERNAL_BUILD
    if (DebugDrawNavMeshFlag)
        DebugDrawRecast(&RecastDebugDrawer, DRAWMODE_NAVMESH);
    if (DebugDrawEnemyPathingFlag)
        DebugDrawFollowPath(RecastDebugDrawer.SupportRenderer);
#endif // INTERNAL_BUILD

#ifdef JPH_DEBUG_RENDERER
    if (DebugDrawLevelColliderFlag)
    {
        Physics.BodyInterface->GetShape(GameState->LevelColliderBodyId)->Draw(JoltDebugDrawer,
            Physics.BodyInterface->GetCenterOfMassTransform(GameState->LevelColliderBodyId),
            JPH::Vec3::sReplicate(1.0f), JPH::Color(0,255,0,60), true, false);
    }

    if (DebugDrawEnemyCollidersFlag)
        DebugDrawEnemyColliders(JoltDebugDrawer);

    if (DebugDrawProjectileCollidersFlag)
    {
        for (size_t i = 0; i < LiveProjectiles.lenu(); ++i)
        {
            projectile_t P = LiveProjectiles[i];
            Physics.BodyInterface->GetShape(P.BodyId)->Draw(JoltDebugDrawer,
                Physics.BodyInterface->GetCenterOfMassTransform(P.BodyId),
                JPH::Vec3::sReplicate(1.0f), JPH::Color(255,0,0,255), false, true);
        }
    }
#endif // JPH_DEBUG_RENDERER
}

void RequestDrawGame()
{
    constexpr u32 MaxNumTexturedLitModels = 128;
    constexpr u32 MaxNumSkinnedModels = 128;

    // Textured Lit Models
    GameState->TexturedLitRenderData = fixed_array<textured_lit_drawinfo>(
        MaxNumTexturedLitModels, MemoryType::Frame);
    DrawWeaponModel(GameState);

    // Skinned Models
    GameState->SMRenderData = fixed_array<sm_drawinfo>(
        MaxNumSkinnedModels, MemoryType::Frame);
    for (u32 i = 0; i < EnemySystem.Enemies.lenu(); ++i)
    {
        enemy_t& Enemy = EnemySystem.Enemies[i];
        if (!(Enemy.Flags & EnemyFlag_Active))
            continue;

        sm_drawinfo SMDrawInfo;
        FillSkinnedModelDrawInfo(
            &SMDrawInfo,
            GameState,
            Enemy.Position,//TODO(Kevin): Centroid
            Enemy.Position,
            Enemy.Orientation,
            Enemy.Animator,
            Assets.Model_Attacker);
        GameState->SMRenderData.put(SMDrawInfo);
    }

    // Projectile Instancing
    InstanceProjectilesForDrawing(GameState);

    // Particles
    GameState->PQuadBuf.setlen(0);
    vec3 QuadDirection = -GameState->Player.PlayerCam.Direction;
    AssembleParticleQuads(GameState->BloodParticles, QuadDirection, GameState->PQuadBuf);

#if INTERNAL_BUILD
    DebugDrawGame();
#endif // INTERNAL_BUILD

    float AspectRatio = 
        float(GameState->AppState->BackBufferWidth) / 
        float(GameState->AppState->BackBufferHeight);
    float FOVX = 90.f;
    float FOVYRad = HorizontalFOVToVerticalFOV_RadianToRadian(FOVX*GM_DEG2RAD, AspectRatio);
    GameState->ClipFromView = ProjectionMatrixPerspective(FOVYRad,
        AspectRatio, GAMEPROJECTION_NEARCLIP, GAMEPROJECTION_FARCLIP);
    GameState->ViewFromWorld = GameState->Player.PlayerCam.ViewFromWorldMatrix();
    GameState->ClipFromWorld = GameState->ClipFromView * GameState->ViewFromWorld;

    RenderGameState(GameState);
}

void CreateAndRegisterLevelCollider(game_state *GameState)
{
    static JPH::TriangleList LevelColliderTriangles;
    LevelColliderTriangles.clear();
    int LoadingLevelColliderPointsIterator = 0;
    for (u32 ColliderIndex = 0; 
        ColliderIndex < GameState->LoadingLevelColliderSpans.lenu(); 
        ++ColliderIndex)
    {
        u32 Span = GameState->LoadingLevelColliderSpans[ColliderIndex];
        vec3 *PointCloudPtr = &GameState->LoadingLevelColliderPoints[LoadingLevelColliderPointsIterator];

        vec3 First = PointCloudPtr[0];
        for (u32 i = 2; i < Span; ++i)
        {
            vec3 Second = PointCloudPtr[i-1];
            vec3 Third = PointCloudPtr[i];

            JPH::Float3 jph_first(ToJoltUnit(First.x), ToJoltUnit(First.y), ToJoltUnit(First.z));
            JPH::Float3 jph_second(ToJoltUnit(Second.x), ToJoltUnit(Second.y), ToJoltUnit(Second.z));
            JPH::Float3 jph_third(ToJoltUnit(Third.x), ToJoltUnit(Third.y), ToJoltUnit(Third.z));

            LevelColliderTriangles.push_back(JPH::Triangle(jph_first, jph_second, jph_third));
        }

        LoadingLevelColliderPointsIterator += Span;
    }
    JPH::MeshShapeSettings LevelColliderSettings = JPH::MeshShapeSettings(LevelColliderTriangles);
    LevelColliderSettings.mBuildQuality = JPH::MeshShapeSettings::EBuildQuality::FavorRuntimePerformance;
    LevelColliderSettings.SetEmbedded();

    // Create the shape
    JPH::ShapeSettings::ShapeResult LevelShapeResult = LevelColliderSettings.Create();
    JPH::ShapeRefC LevelShape = LevelShapeResult.Get(); // We don't expect an error here, but you can check floor_shape_result for HasError() / GetError()
    if(LevelShapeResult.HasError())
    {
        LogMessage("%s", LevelShapeResult.GetError().c_str());
    }

    // Create the settings for the body itself. Note that here you can also set other properties like the restitution / friction.
    JPH::BodyCreationSettings LevelBodySettings(LevelShape, JPH::RVec3(0.0, 0.0, 0.0), 
        JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::STATIC);
    LevelBodySettings.mEnhancedInternalEdgeRemoval = true;

    // Create the actual rigid body
    JPH::Body *LevelCollider = Physics.BodyInterface->CreateBody(LevelBodySettings); // Note that if we run out of bodies this can return nullptr

    GameState->LevelColliderBodyId = LevelCollider->GetID();

    // Add it to the world
    // NOTE(Kevin 2025-01-30): Why is this DontActivate?
    Physics.BodyInterface->AddBody(LevelCollider->GetID(), JPH::EActivation::DontActivate);

    // Optional step: Before starting the physics simulation you can optimize the broad phase. This improves collision detection performance (it's pointless here because we only have 2 bodies).
    // You should definitely not call this every frame or when e.g. streaming in a new level section as it is an expensive operation.
    // Instead insert all new objects in batches instead of 1 at a time to keep the broad phase efficient.
    Physics.PhysicsSystem->OptimizeBroadPhase();
}
