
Mix_Chunk *sfx_Jump;
ModelGLTF Model_Knight;

// extern
std::vector<face_batch_t> GameLevelFaceBatches;
bool GameLoopCanRun = true;

// internal
bool LevelLoaded = false;
physics_t Physics;
player_t Player;
mat4 GameViewMatrix;
JPH::BodyID LevelColliderBodyId; 

void InitializeGame()
{
    // testing stuff here
    sfx_Jump = Mixer_LoadChunk(wd_path("gunshot-37055.ogg").c_str());
    LoadModelGLTF2Bin(&Model_Knight, wd_path("models/knight.glb").c_str());

    Physics.Initialize();

    CreateAndRegisterPlayerPhysicsController();
}

void DestroyGame()
{
    Physics.Destroy();
}

void LoadLevel(const char *MapPath)
{
    ASSERT(Physics.PhysicsSystem);

    SDL_SetRelativeMouseMode(SDL_TRUE);

    if (LevelLoaded)
        UnloadPreviousLevel();

    map_load_result_t MapLoadResult = LoadGameMap(MapPath);
    if (MapLoadResult.Success == false)
    {
        LogError("failed to load game map");
        return;
    }

    CreateAndRegisterLevelCollider();
    bool yo = CreateRecastNavMesh();

    Player.mCharacter->SetPositionAndRotation(
        ToJoltVec3(MapLoadResult.PlayerStartPosition),
        ToJoltQuat(EulerToQuat(MapLoadResult.PlayerStartRotation)));
    // Apply to Jolt Character or my camera rotation?

    LevelLoaded = true;
}

void UnloadPreviousLevel()
{
    if (!LevelColliderBodyId.IsInvalid())
    {
        Physics.BodyInterface->RemoveBody(LevelColliderBodyId);
        Physics.BodyInterface->DestroyBody(LevelColliderBodyId);
    }

    LevelLoaded = false;
}

void PrePhysicsTick()
{

    // CALCULATE PLAYER FACING DIRECTION
    float camYawDelta = MouseDelta.x * 0.085f;
    float camPitchDelta = MouseDelta.y * 0.085f;
    Player.CameraRotation.y -= camYawDelta;
    Player.CameraRotation.z -= camPitchDelta;
    if (Player.CameraRotation.z > 89.f)
        Player.CameraRotation.z = 89.f;
    if (Player.CameraRotation.z < -89.f)
        Player.CameraRotation.z = -89.f;
    Player.CameraDirection = Normalize(OrientationToDirection(EulerToQuat(Player.CameraRotation * GM_DEG2RAD)));
    Player.CameraRight = Normalize(Cross(Player.CameraDirection, GM_UP_VECTOR));
    Player.CameraUp = Normalize(Cross(Player.CameraRight, Player.CameraDirection));
    Player.WalkDirectionRight = Player.CameraRight;
    Player.WalkDirectionForward = Normalize(Cross(GM_UP_VECTOR, Player.WalkDirectionRight));

    // PLAYER MOVE
    vec3 desiredMoveDirection;
    if (KeysCurrent[SDL_SCANCODE_W])
        desiredMoveDirection += Player.WalkDirectionForward;
    if (KeysCurrent[SDL_SCANCODE_A])
        desiredMoveDirection += -Player.WalkDirectionRight;
    if (KeysCurrent[SDL_SCANCODE_S])
        desiredMoveDirection += -Player.WalkDirectionForward;
    if (KeysCurrent[SDL_SCANCODE_D])
        desiredMoveDirection += Player.WalkDirectionRight;
    

    // Cancel movement in opposite direction of normal when touching something we can't walk up
    JPH::Vec3 movement_direction = JPH::Vec3(desiredMoveDirection.x, desiredMoveDirection.y, desiredMoveDirection.z);
    JPH::Character::EGroundState ground_state = Player.mCharacter->GetGroundState();
    if (ground_state == JPH::Character::EGroundState::OnSteepGround || 
        ground_state == JPH::Character::EGroundState::NotSupported)
    {
        JPH::Vec3 normal = Player.mCharacter->GetGroundNormal();
        normal.SetY(0.0f);
        float dot = normal.Dot(movement_direction);
        if (dot < 0.0f)
            movement_direction -= (dot * normal) / normal.LengthSq();
    }

    //// Stance switch
    //if (inSwitchStance)
    //    Player.mCharacter->SetShape(Player.mCharacter->GetShape() == mStandingShape ? mCrouchingShape : mStandingShape, 1.5f * mPhysicsSystem->GetPhysicsSettings().mPenetrationSlop);
    const float sCharacterSpeed = 6.0f*32.f;
    const float sJumpSpeed = 4.0f*32.f;
    if (/*sControlMovementDuringJump || */ Player.mCharacter->IsSupported())
    {
        // Update velocity
        JPH::Vec3 current_velocity = Player.mCharacter->GetLinearVelocity();
        // try printing magnitude of current velocity
        JPH::Vec3 desired_velocity = sCharacterSpeed * movement_direction;
        if (!desired_velocity.IsNearZero() || current_velocity.GetY() < 0.0f || !Player.mCharacter->IsSupported())
            desired_velocity.SetY(current_velocity.GetY());
        JPH::Vec3 new_velocity = 0.75f * current_velocity + 0.25f * desired_velocity;

        // Jump
        if (KeysPressed[SDL_SCANCODE_SPACE] && ground_state == JPH::Character::EGroundState::OnGround)
        {
            new_velocity += JPH::Vec3(0, sJumpSpeed, 0);

            // static int channelrotationtesting = 0;
            // Mix_VolumeChunk(sfx_Jump, 48);
            // Mix_PlayChannel(channelrotationtesting++%3, sfx_Jump, 0);
        }

        // Update the velocity
        Player.mCharacter->SetLinearVelocity(new_velocity);
    }

}

void PostPhysicsTick()
{
    static const float cCollisionTolerance = 0.05f;
    Player.mCharacter->PostSimulation(cCollisionTolerance);

    JPH::RVec3 cpos = Player.mCharacter->GetPosition();
    // LogMessage("character pos %f, %f, %f", cpos.GetX(), cpos.GetY(), cpos.GetZ());

    Player.Root.x = cpos.GetX();
    Player.Root.y = cpos.GetY();
    Player.Root.z = cpos.GetZ();


    // PLAYER CAMERA 
    vec3 CameraPosOffsetFromRoot = vec3(0,40,0);
    vec3 CameraPosition = Player.Root + CameraPosOffsetFromRoot;
    // LogMessage("pos %f, %f, %f", playerControllerRoot.x, playerControllerRoot.y, playerControllerRoot.z);
    // LogMessage("dir y %f, z %f\n", cameraRotation.y, cameraRotation.z);

    static float camLean = 0.f;
    static float desiredCamLean = 0.f;
    const float camLeanSpeed = 15;
    const float maxCamLean = 0.07f;
    desiredCamLean = 0.f;
    if(KeysCurrent[SDL_SCANCODE_D])
        desiredCamLean += maxCamLean;
    if(KeysCurrent[SDL_SCANCODE_A])
        desiredCamLean += -maxCamLean;
    camLean = Lerp(camLean, desiredCamLean, DeltaTime * camLeanSpeed);

    quat fromto = RotationFromTo(Player.CameraUp, Player.CameraRight);
    quat sle = Slerp(quat(), fromto, camLean);
    vec3 CameraUpWithSway = RotateVector(Player.CameraUp, sle);
    float dot = Dot(Normalize(Cross(CameraUpWithSway, Player.CameraRight)), Player.CameraDirection);
    if (dot < 0.99f)
        printf("bad cam up %f\n", dot);
    GameViewMatrix = ViewMatrixLookAt(CameraPosition, CameraPosition + Player.CameraDirection, CameraUpWithSway);
}

void DoGameLoop()
{
    if (!GameLoopCanRun) 
        return;

    // pre physics tick
    PrePhysicsTick();

    // physics tick
    Physics.Tick();

    // post physics tick
    PostPhysicsTick();

    // Do animation loop

    // Do render loop
    RenderGameLayer();

    UpdateGameGUI();
}

void UpdateGameGUI()
{
    // temp crosshair
    ivec2 guiwh = ivec2(RenderTargetGUI.width, RenderTargetGUI.height);
    GUI::PrimitivePanel(GUI::UIRect(guiwh.x / 2 - 3, guiwh.y / 2 - 3, 6, 6), vec4(0, 0, 0, 1));
    GUI::PrimitivePanel(GUI::UIRect(guiwh.x / 2 - 2, guiwh.y / 2 - 2, 4, 4), vec4(1, 1, 1, 1));
}

void DoDebugDrawRecast();
void RenderGameLayer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, RenderTargetGame.fbo);
    glViewport(0, 0, RenderTargetGame.width, RenderTargetGame.height);
    glClearColor(0.674f, 0.847f, 1.0f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //RGBHEXTO1(0x6495ed), 1.f);//(RGB255TO1(211, 203, 190), 1.f);//(0.674f, 0.847f, 1.0f, 1.f); //RGB255TO1(46, 88, 120)
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_DEPTH_TEST);

    float aspectratio = float(BackbufferWidth) / float(BackbufferHeight);
    float fovy = HorizontalFOVToVerticalFOV_RadianToRadian(90.f*GM_DEG2RAD, aspectratio);
    mat4 perspectiveMatrix = ProjectionMatrixPerspective(fovy, aspectratio, GAMEPROJECTION_NEARCLIP, GAMEPROJECTION_FARCLIP);
    mat4 viewMatrix = GameViewMatrix;

    UseShader(GameLevelShader);
    glEnable(GL_CULL_FACE);

    GLBindMatrix4fv(GameLevelShader, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(GameLevelShader, "viewMatrix", 1, viewMatrix.ptr());

    for (size_t i = 0; i < GameLevelFaceBatches.size(); ++i)
    {
        face_batch_t fb = GameLevelFaceBatches.at(i);
        RenderFaceBatch(&GameLevelShader, &fb);
    }

    // modelMatrix = TranslationMatrix(enemy0.root) * RotationMatrix(DirectionToOrientation(enemy0.facing));
    // GLBindMatrix4fv(gameLevelShader, "modelMatrix", 1, modelMatrix.ptr());
    // RenderModelGLTF(model_Knight);

    // PRIMITIVES    
    static bool DoPrimitivesDepthTest = false;
    if (KeysPressed[SDL_SCANCODE_X])
        DoPrimitivesDepthTest = !DoPrimitivesDepthTest;
    if (DoPrimitivesDepthTest)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    SupportRenderer.FlushPrimitives(&perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));

    // DoDebugDrawRecast();
}

#include <Recast.h>
#include <DebugDraw.h>
#include <RecastDebugDraw.h>
// #include <DetourNavMesh.h>

#ifdef __GNUC__
#include <stdint.h>
typedef int64_t TimeVal;
#else
typedef __int64 TimeVal;
#endif

TimeVal getPerfTime();
int getPerfTimeUsec(const TimeVal duration);

TimeVal getPerfTime()
{
    __int64 count;
    QueryPerformanceCounter((LARGE_INTEGER*)&count);
    return count;
}

int getPerfTimeUsec(const TimeVal duration)
{
    static __int64 freq = 0;
    if (freq == 0)
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    return (int)(duration*1000000 / freq);
}

class BuildContext : public rcContext
{
    TimeVal m_startTime[RC_MAX_TIMERS];
    TimeVal m_accTime[RC_MAX_TIMERS];

    static const int MAX_MESSAGES = 1000;
    const char* m_messages[MAX_MESSAGES];
    int m_messageCount;
    static const int TEXT_POOL_SIZE = 8000;
    char m_textPool[TEXT_POOL_SIZE];
    int m_textPoolSize;
    
public:
    BuildContext();
    
    /// Dumps the log to stdout.
    void dumpLog(const char* format, ...);
    /// Returns number of log messages.
    int getLogCount() const;
    /// Returns log message text.
    const char* getLogText(const int i) const;
    
protected:  
    /// Virtual functions for custom implementations.
    ///@{
    virtual void doResetLog();
    virtual void doLog(const rcLogCategory category, const char* msg, const int len);
    virtual void doResetTimers();
    virtual void doStartTimer(const rcTimerLabel label);
    virtual void doStopTimer(const rcTimerLabel label);
    virtual int doGetAccumulatedTime(const rcTimerLabel label) const;
    ///@}
};

BuildContext::BuildContext() 
    : m_messageCount(0)
    , m_textPoolSize(0)
{
    memset(m_messages, 0, sizeof(char*) * MAX_MESSAGES);

    resetTimers();
}

// Virtual functions for custom implementations.
void BuildContext::doResetLog()
{
    m_messageCount = 0;
    m_textPoolSize = 0;
}

void BuildContext::doLog(const rcLogCategory category, const char* msg, const int len)
{
    if (!len) return;
    if (m_messageCount >= MAX_MESSAGES)
        return;
    char* dst = &m_textPool[m_textPoolSize];
    int n = TEXT_POOL_SIZE - m_textPoolSize;
    if (n < 2)
        return;
    char* cat = dst;
    char* text = dst+1;
    const int maxtext = n-1;
    // Store category
    *cat = (char)category;
    // Store message
    const int count = rcMin(len+1, maxtext);
    memcpy(text, msg, count);
    text[count-1] = '\0';
    m_textPoolSize += 1 + count;
    m_messages[m_messageCount++] = dst;
}

void BuildContext::doResetTimers()
{
    for (int i = 0; i < RC_MAX_TIMERS; ++i)
        m_accTime[i] = -1;
}

void BuildContext::doStartTimer(const rcTimerLabel label)
{
    m_startTime[label] = getPerfTime();
}

void BuildContext::doStopTimer(const rcTimerLabel label)
{
    const TimeVal endTime = getPerfTime();
    const TimeVal deltaTime = endTime - m_startTime[label];
    if (m_accTime[label] == -1)
        m_accTime[label] = deltaTime;
    else
        m_accTime[label] += deltaTime;
}

int BuildContext::doGetAccumulatedTime(const rcTimerLabel label) const
{
    return getPerfTimeUsec(m_accTime[label]);
}

void BuildContext::dumpLog(const char* format, ...)
{
    // Print header.
    va_list ap;
    va_start(ap, format);
    vprintf(format, ap);
    va_end(ap);
    printf("\n");
    
    // Print messages
    const int TAB_STOPS[4] = { 28, 36, 44, 52 };
    for (int i = 0; i < m_messageCount; ++i)
    {
        const char* msg = m_messages[i]+1;
        int n = 0;
        while (*msg)
        {
            if (*msg == '\t')
            {
                int count = 1;
                for (int j = 0; j < 4; ++j)
                {
                    if (n < TAB_STOPS[j])
                    {
                        count = TAB_STOPS[j] - n;
                        break;
                    }
                }
                while (--count)
                {
                    putchar(' ');
                    n++;
                }
            }
            else
            {
                putchar(*msg);
                n++;
            }
            msg++;
        }
        putchar('\n');
    }
}

int BuildContext::getLogCount() const
{
    return m_messageCount;
}

const char* BuildContext::getLogText(const int i) const
{
    return m_messages[i]+1;
}

// class DebugDrawGL : public duDebugDraw
// {
// public:
//     virtual void depthMask(bool state);
//     virtual void texture(bool state);
//     virtual void begin(duDebugDrawPrimitives prim, float size = 1.0f);
//     virtual void vertex(const float* pos, unsigned int color);
//     virtual void vertex(const float x, const float y, const float z, unsigned int color);
//     virtual void vertex(const float* pos, unsigned int color, const float* uv);
//     virtual void vertex(const float x, const float y, const float z, unsigned int color, const float u, const float v);
//     virtual void end();
// };


static BuildContext *m_ctx;
static rcConfig m_cfg;
static rcHeightfield *m_solid;
static unsigned char *m_triareas;
static rcCompactHeightfield *m_chf;
static rcContourSet *m_cset;
static rcPolyMesh *m_pmesh;
static rcPolyMeshDetail* m_dmesh;

enum SamplePartitionType
{
    SAMPLE_PARTITION_WATERSHED,
    SAMPLE_PARTITION_MONOTONE,
    SAMPLE_PARTITION_LAYERS
};

bool CreateRecastNavMesh()
{
    LogMessage("Building Recast NavMesh");

    BuildContext ctx;
    m_ctx = &ctx;

    std::vector<int> LevelColliderTriangles;
    int LoadingLevelColliderPointsIterator = 0;
    for (u32 ColliderIndex = 0; ColliderIndex < (u32)LoadingLevelColliderSpans.size(); ++ColliderIndex)
    {
        u32 Span = LoadingLevelColliderSpans[ColliderIndex];
        vec3 *PointCloudPtr = &LoadingLevelColliderPoints[LoadingLevelColliderPointsIterator];

        for (u32 i = 2; i < Span; ++i)
        {
            int FirstIndex = LoadingLevelColliderPointsIterator;
            int SecondIndex = LoadingLevelColliderPointsIterator + i - 1;
            int ThirdIndex = LoadingLevelColliderPointsIterator + i;

            vec3 Second = PointCloudPtr[i-1];
            vec3 Third = PointCloudPtr[i];

            LevelColliderTriangles.push_back(FirstIndex);
            LevelColliderTriangles.push_back(SecondIndex);
            LevelColliderTriangles.push_back(ThirdIndex);
        }

        LoadingLevelColliderPointsIterator += Span;
    }
    ASSERT(LevelColliderTriangles.size() % 3 == 0);

    // min max
    vec3 min = vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    vec3 max = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (u32 i = 0; i < (u32)LoadingLevelColliderPoints.size(); ++i)
    {
        vec3 point = LoadingLevelColliderPoints[i];
        min.x = GM_min(min.x, point.x);
        min.y = GM_min(min.y, point.y);
        min.z = GM_min(min.z, point.z);
        max.x = GM_max(max.x, point.x);
        max.y = GM_max(max.y, point.y);
        max.z = GM_max(max.z, point.z);
    }

    const float* bmin = (float*)&min;//m_geom->getNavMeshBoundsMin();
    const float* bmax = (float*)&max;//m_geom->getNavMeshBoundsMax();
    const float* verts = (float*)LoadingLevelColliderPoints.data();//m_geom->getMesh()->getVerts();
    const int nverts = (int)LoadingLevelColliderPoints.size()*3;//m_geom->getMesh()->getVertCount();
    const int* tris = (int*)LevelColliderTriangles.data();
    const int ntris = (int)LevelColliderTriangles.size() / 3;

    // NOTE(Kevin): Need to tweak these configs to fit my levels
    //              5 cell size and 2 cell height seems to work ok
    
    //
    // Step 1. Initialize build config.
    //
    
    // Init build configuration from GUI
    memset(&m_cfg, 0, sizeof(m_cfg));
    /// The xz-plane cell size to use for fields. [Limit: > 0] [Units: wu] 
    m_cfg.cs = 3.f;//8;
    /// The y-axis cell size to use for fields. [Limit: > 0] [Units: wu]
    m_cfg.ch = 2.f;//6;
    /// The maximum slope that is considered walkable. [Limits: 0 <= value < 90] [Units: Degrees] 
    m_cfg.walkableSlopeAngle = 45;
    /// Minimum floor to 'ceiling' height that will still allow the floor area to 
    /// be considered walkable. [Limit: >= 3] [Units: vx] 
    m_cfg.walkableHeight = (int)ceilf(2.0f / m_cfg.ch);//6;//(int)ceilf(m_agentHeight / m_cfg.ch);
    /// Maximum ledge height that is considered to still be traversable. [Limit: >=0] [Units: vx]
    m_cfg.walkableClimb = (int)floorf(0.9f / m_cfg.ch);//6;//(int)floorf(m_agentMaxClimb / m_cfg.ch);
    /// The distance to erode/shrink the walkable area of the heightfield away from 
    /// obstructions.  [Limit: >=0] [Units: vx] 
    m_cfg.walkableRadius = (int)ceilf(0.6f / m_cfg.cs);//2;//(int)ceilf(m_agentRadius / m_cfg.cs);
    /// The maximum allowed length for contour edges along the border of the mesh. [Limit: >=0] [Units: vx] 
    m_cfg.maxEdgeLen = (int)(12.f / m_cfg.cs);//40;//(int)(m_edgeMaxLen / m_cellSize);
    /// The maximum distance a simplified contour's border edges should deviate 
    /// the original raw contour. [Limit: >=0] [Units: vx]
    m_cfg.maxSimplificationError = 1.3f;//35;//m_edgeMaxError;
    /// The minimum number of cells allowed to form isolated island areas. [Limit: >=0] [Units: vx] 
    // Note(Kevin): 27 because my 8 cell size / 0.3 from the sample = 27
    m_cfg.minRegionArea = (int)rcSqr(8);//(int)rcSqr(8*27);//(int)rcSqr(m_regionMinSize);      // Note: area = size*size
    /// Any regions with a span count smaller than this value will, if possible, 
    /// be merged with larger regions. [Limit: >=0] [Units: vx] 
    m_cfg.mergeRegionArea = (int)rcSqr(20);//(int)rcSqr(20*27);//(int)rcSqr(m_regionMergeSize);  // Note: area = size*size
    /// The maximum number of vertices allowed for polygons generated during the 
    /// contour to polygon conversion process. [Limit: >= 3] 
    m_cfg.maxVertsPerPoly = 6;//(int)m_vertsPerPoly;
    /// Sets the sampling distance to use when generating the detail mesh.
    /// (For height detail only.) [Limits: 0 or >= 0.9] [Units: wu] 
    m_cfg.detailSampleDist = m_cfg.cs * 6.f;//8.f;//m_detailSampleDist < 0.9f ? 0 : m_cellSize * m_detailSampleDist;
    /// The maximum distance the detail mesh surface should deviate from heightfield
    /// data. (For height detail only.) [Limit: >=0] [Units: wu] 
    m_cfg.detailSampleMaxError = m_cfg.ch * 1.f;//m_cellHeight * m_detailSampleMaxError;
    
    // Set the area where the navigation will be build.
    // Here the bounds of the input mesh are used, but the
    // area could be specified by an user defined box, etc.
    rcVcopy(m_cfg.bmin, bmin);
    rcVcopy(m_cfg.bmax, bmax);
    rcCalcGridSize(m_cfg.bmin, m_cfg.bmax, m_cfg.cs, &m_cfg.width, &m_cfg.height);

    // // Reset build times gathering.
    // m_ctx->resetTimers();

    // // Start the build process. 
    // m_ctx->startTimer(RC_TIMER_TOTAL);
    
    // m_ctx->log(RC_LOG_PROGRESS, "Building navigation:");
    // m_ctx->log(RC_LOG_PROGRESS, " - %d x %d cells", m_cfg.width, m_cfg.height);
    // m_ctx->log(RC_LOG_PROGRESS, " - %.1fK verts, %.1fK tris", nverts/1000.0f, ntris/1000.0f);

    //
    // Step 2. Rasterize input polygon soup.
    //
    
    // Allocate voxel heightfield where we rasterize our input data to.
    m_solid = rcAllocHeightfield();
    if (!m_solid)
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Out of memory 'solid'.");
        return false;
    }
    if (!rcCreateHeightfield(m_ctx, *m_solid, m_cfg.width, m_cfg.height, m_cfg.bmin, m_cfg.bmax, m_cfg.cs, m_cfg.ch))
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Could not create solid heightfield.");
        return false;
    }
    
    // Allocate array that can hold triangle area types.
    // If you have multiple meshes you need to process, allocate
    // and array which can hold the max number of triangles you need to process.
    m_triareas = new unsigned char[ntris];
    if (!m_triareas)
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Out of memory 'm_triareas' (%d).", ntris);
        return false;
    }
    
    // Find triangles which are walkable based on their slope and rasterize them.
    // If your input data is multiple meshes, you can transform them here, calculate
    // the are type for each of the meshes and rasterize them.
    memset(m_triareas, 0, ntris*sizeof(unsigned char));
    rcMarkWalkableTriangles(m_ctx, m_cfg.walkableSlopeAngle, verts, nverts, tris, ntris, m_triareas);
    if (!rcRasterizeTriangles(m_ctx, verts, nverts, tris, m_triareas, ntris, *m_solid, m_cfg.walkableClimb))
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Could not rasterize triangles.");
        return false;
    }

    const bool m_keepInterResults = false;
    const bool m_filterLowHangingObstacles = true;
    const bool m_filterLedgeSpans = true;
    const bool m_filterWalkableLowHeightSpans = true;
    if (!m_keepInterResults)
    {
        delete [] m_triareas;
        m_triareas = 0;
    }

    //
    // Step 3. Filter walkable surfaces.
    //
    
    // Once all geometry is rasterized, we do initial pass of filtering to
    // remove unwanted overhangs caused by the conservative rasterization
    // as well as filter spans where the character cannot possibly stand.
    if (m_filterLowHangingObstacles)
        rcFilterLowHangingWalkableObstacles(m_ctx, m_cfg.walkableClimb, *m_solid);
    if (m_filterLedgeSpans)
        rcFilterLedgeSpans(m_ctx, m_cfg.walkableHeight, m_cfg.walkableClimb, *m_solid);
    if (m_filterWalkableLowHeightSpans)
        rcFilterWalkableLowHeightSpans(m_ctx, m_cfg.walkableHeight, *m_solid);

    //
    // Step 4. Partition walkable surface to simple regions.
    //

    // Compact the heightfield so that it is faster to handle from now on.
    // This will result more cache coherent data as well as the neighbours
    // between walkable cells will be calculated.
    m_chf = rcAllocCompactHeightfield();
    if (!m_chf)
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Out of memory 'chf'.");
        return false;
    }
    if (!rcBuildCompactHeightfield(m_ctx, m_cfg.walkableHeight, m_cfg.walkableClimb, *m_solid, *m_chf))
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Could not build compact data.");
        return false;
    }
    
    if (!m_keepInterResults)
    {
        rcFreeHeightField(m_solid);
        m_solid = 0;
    }
        
    // Erode the walkable area by agent radius.
    if (!rcErodeWalkableArea(m_ctx, m_cfg.walkableRadius, *m_chf))
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Could not erode.");
        return false;
    }

    // Partition the heightfield so that we can use simple algorithm later to triangulate the walkable areas.
    // There are 3 partitioning methods, each with some pros and cons:
    // 1) Watershed partitioning
    //   - the classic Recast partitioning
    //   - creates the nicest tessellation
    //   - usually slowest
    //   - partitions the heightfield into nice regions without holes or overlaps
    //   - the are some corner cases where this method creates produces holes and overlaps
    //      - holes may appear when a small obstacles is close to large open area (triangulation can handle this)
    //      - overlaps may occur if you have narrow spiral corridors (i.e stairs), this make triangulation to fail
    //   * generally the best choice if you precompute the navmesh, use this if you have large open areas
    // 2) Monotone partitioning
    //   - fastest
    //   - partitions the heightfield into regions without holes and overlaps (guaranteed)
    //   - creates long thin polygons, which sometimes causes paths with detours
    //   * use this if you want fast navmesh generation
    // 3) Layer partitoining
    //   - quite fast
    //   - partitions the heighfield into non-overlapping regions
    //   - relies on the triangulation code to cope with holes (thus slower than monotone partitioning)
    //   - produces better triangles than monotone partitioning
    //   - does not have the corner cases of watershed partitioning
    //   - can be slow and create a bit ugly tessellation (still better than monotone)
    //     if you have large open areas with small obstacles (not a problem if you use tiles)
    //   * good choice to use for tiled navmesh with medium and small sized tiles
    
    const SamplePartitionType m_partitionType = SAMPLE_PARTITION_WATERSHED;
    if (m_partitionType == SAMPLE_PARTITION_WATERSHED)
    {
        // Prepare for region partitioning, by calculating distance field along the walkable surface.
        if (!rcBuildDistanceField(m_ctx, *m_chf))
        {
            m_ctx->log(RC_LOG_ERROR, "buildNavigation: Could not build distance field.");
            return false;
        }
        
        // Partition the walkable surface into simple regions without holes.
        if (!rcBuildRegions(m_ctx, *m_chf, 0, m_cfg.minRegionArea, m_cfg.mergeRegionArea))
        {
            m_ctx->log(RC_LOG_ERROR, "buildNavigation: Could not build watershed regions.");
            return false;
        }
    }
    else if (m_partitionType == SAMPLE_PARTITION_MONOTONE)
    {
        // Partition the walkable surface into simple regions without holes.
        // Monotone partitioning does not need distancefield.
        if (!rcBuildRegionsMonotone(m_ctx, *m_chf, 0, m_cfg.minRegionArea, m_cfg.mergeRegionArea))
        {
            m_ctx->log(RC_LOG_ERROR, "buildNavigation: Could not build monotone regions.");
            return false;
        }
    }
    else // SAMPLE_PARTITION_LAYERS
    {
        // Partition the walkable surface into simple regions without holes.
        if (!rcBuildLayerRegions(m_ctx, *m_chf, 0, m_cfg.minRegionArea))
        {
            m_ctx->log(RC_LOG_ERROR, "buildNavigation: Could not build layer regions.");
            return false;
        }
    }
    
    //
    // Step 5. Trace and simplify region contours.
    //
    
    // Create contours.
    m_cset = rcAllocContourSet();
    if (!m_cset)
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Out of memory 'cset'.");
        return false;
    }
    if (!rcBuildContours(m_ctx, *m_chf, m_cfg.maxSimplificationError, m_cfg.maxEdgeLen, *m_cset))
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Could not create contours.");
        return false;
    }
    
    //
    // Step 6. Build polygons mesh from contours.
    //
    
    // Build polygon navmesh from the contours.
    m_pmesh = rcAllocPolyMesh();
    if (!m_pmesh)
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Out of memory 'pmesh'.");
        return false;
    }
    if (!rcBuildPolyMesh(m_ctx, *m_cset, m_cfg.maxVertsPerPoly, *m_pmesh))
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Could not triangulate contours.");
        return false;
    }

    //
    // Step 7. Create detail mesh which allows to access approximate height on each polygon.
    //
    
    m_dmesh = rcAllocPolyMeshDetail();
    if (!m_dmesh)
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Out of memory 'pmdtl'.");
        return false;
    }

    if (!rcBuildPolyMeshDetail(m_ctx, *m_pmesh, *m_chf, m_cfg.detailSampleDist, m_cfg.detailSampleMaxError, *m_dmesh))
    {
        m_ctx->log(RC_LOG_ERROR, "buildNavigation: Could not build detail mesh.");
        return false;
    }

    if (!m_keepInterResults)
    {
        rcFreeCompactHeightfield(m_chf);
        m_chf = 0;
        rcFreeContourSet(m_cset);
        m_cset = 0;
    }

    // At this point the navigation mesh data is ready, you can access it from m_pmesh.
    // See duDebugDrawPolyMesh or dtCreateNavMeshData as examples how to access the data.


    return true;
}

void DoDebugDrawRecast()
{
    // keep in mind I can write my own duDebugDraw subclass that uses
    // OpenGL 3+

    const rcPolyMesh& mesh = *m_pmesh;
    const int nvp = mesh.nvp;
    const float cs = mesh.cs;
    const float ch = mesh.ch;
    const float* orig = mesh.bmin;
    
    // dd->begin(DU_DRAW_TRIS);

    static std::vector<vec3> VertexBuffer;
    VertexBuffer.clear();
    
    for (int i = 0; i < mesh.npolys; ++i)
    {
        const unsigned short* p = &mesh.polys[i*nvp*2];
        const unsigned char area = mesh.areas[i];
        
        // unsigned int color;
        // if (area == RC_WALKABLE_AREA)
        //     color = duRGBA(0,192,255,64);
        // else if (area == RC_NULL_AREA)
        //     color = duRGBA(0,0,0,64);
        // else
        //     color = dd->areaToCol(area);
        
        unsigned short vi[3];
        for (int j = 2; j < nvp; ++j)
        {
            if (p[j] == RC_MESH_NULL_IDX) break;
            vi[0] = p[0];
            vi[1] = p[j-1];
            vi[2] = p[j];
            for (int k = 0; k < 3; ++k)
            {
                const unsigned short* v = &mesh.verts[vi[k]*3];
                const float x = orig[0] + v[0]*cs;
                const float y = orig[1] + (v[1]+1)*ch;
                const float z = orig[2] + v[2]*cs;
                // dd->vertex(x,y,z, color);
                VertexBuffer.push_back(vec3(x,y,z));
                VertexBuffer.push_back(vec3(1.0,0.0,1.0));
            }
        }
    }

    float aspectratio = float(BackbufferWidth) / float(BackbufferHeight);
    float fovy = HorizontalFOVToVerticalFOV_RadianToRadian(90.f*GM_DEG2RAD, aspectratio);
    mat4 perspectiveMatrix = ProjectionMatrixPerspective(fovy, aspectratio, GAMEPROJECTION_NEARCLIP, GAMEPROJECTION_FARCLIP);
    mat4 viewMatrix = GameViewMatrix;
    SupportRenderer.DrawHandlesVertexArray_GL((float*)VertexBuffer.data(), (u32)VertexBuffer.size()*3,
        perspectiveMatrix.ptr(), viewMatrix.ptr());

    // dd->end();
}

void CreateAndRegisterLevelCollider()
{
    static JPH::TriangleList LevelColliderTriangles;
    LevelColliderTriangles.clear();
    int LoadingLevelColliderPointsIterator = 0;
    for (u32 ColliderIndex = 0; ColliderIndex < (u32)LoadingLevelColliderSpans.size(); ++ColliderIndex)
    {
        u32 Span = LoadingLevelColliderSpans[ColliderIndex];
        vec3 *PointCloudPtr = &LoadingLevelColliderPoints[LoadingLevelColliderPointsIterator];

        vec3 First = PointCloudPtr[0];
        for (u32 i = 2; i < Span; ++i)
        {
            vec3 Second = PointCloudPtr[i-1];
            vec3 Third = PointCloudPtr[i];

            JPH::Float3 jph_first(First.x, First.y, First.z);
            JPH::Float3 jph_second(Second.x, Second.y, Second.z);
            JPH::Float3 jph_third(Third.x, Third.y, Third.z);

            LevelColliderTriangles.push_back(JPH::Triangle(jph_first, jph_second, jph_third));
        }

        LoadingLevelColliderPointsIterator += Span;
    }
    JPH::MeshShapeSettings LevelColliderSettings = JPH::MeshShapeSettings(LevelColliderTriangles);
    LevelColliderSettings.SetEmbedded();

    // Create the shape
    JPH::ShapeSettings::ShapeResult LevelShapeResult = LevelColliderSettings.Create();
    JPH::ShapeRefC LevelShape = LevelShapeResult.Get(); // We don't expect an error here, but you can check floor_shape_result for HasError() / GetError()
    if(LevelShapeResult.HasError())
    {
        LogMessage("%s", LevelShapeResult.GetError().c_str());
    }

    // Create the settings for the body itself. Note that here you can also set other properties like the restitution / friction.
    JPH::BodyCreationSettings LevelBodySettings(LevelShape, JPH::RVec3(0.0, 0.0, 0.0), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::NON_MOVING);
    LevelBodySettings.mEnhancedInternalEdgeRemoval = true;

    // Create the actual rigid body
    JPH::Body *LevelCollider = Physics.BodyInterface->CreateBody(LevelBodySettings); // Note that if we run out of bodies this can return nullptr

    LevelColliderBodyId = LevelCollider->GetID();

    // Add it to the world
    // NOTE(Kevin 2025-01-30): Why is this DontActivate?
    Physics.BodyInterface->AddBody(LevelCollider->GetID(), JPH::EActivation::DontActivate);

    // Optional step: Before starting the physics simulation you can optimize the broad phase. This improves collision detection performance (it's pointless here because we only have 2 bodies).
    // You should definitely not call this every frame or when e.g. streaming in a new level section as it is an expensive operation.
    // Instead insert all new objects in batches instead of 1 at a time to keep the broad phase efficient.
    Physics.PhysicsSystem->OptimizeBroadPhase();
}

void CreateAndRegisterPlayerPhysicsController()
{
    // make the character collider
    static constexpr float cCharacterHeightStanding = 48.f;
    static constexpr float cCharacterRadiusStanding = 8.f;
    JPH::RefConst<JPH::Shape> mStandingShape = JPH::RotatedTranslatedShapeSettings(
        JPH::Vec3(0, 0.5f * cCharacterHeightStanding + cCharacterRadiusStanding, 0), 
        JPH::Quat::sIdentity(), 
        new JPH::CapsuleShape(0.5f * cCharacterHeightStanding, cCharacterRadiusStanding)).Create().Get();

    JPH::Ref<JPH::CharacterSettings> settings = new JPH::CharacterSettings();
    settings->mMaxSlopeAngle = JPH::DegreesToRadians(45.0f);
    settings->mLayer = Layers::MOVING;
    settings->mShape = mStandingShape;
    settings->mFriction = 0.5f;
    settings->mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -cCharacterRadiusStanding); // Accept contacts that touch the lower sphere of the capsule
    Player.mCharacter = new JPH::Character(settings, JPH::RVec3(0,32,0), JPH::Quat::sIdentity(), 0, Physics.PhysicsSystem);
    Player.mCharacter->AddToPhysicsSystem(JPH::EActivation::Activate);
}
