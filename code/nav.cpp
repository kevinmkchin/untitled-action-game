#include "nav.h"

#include <Recast.h>
#include <DetourNavMesh.h>
#include <DetourNavMeshBuilder.h>
#include <DetourNavMeshQuery.h>
#if INTERNAL_BUILD
#include "primitives.h"
#include <RecastDebugDraw.h>
#include <DetourDebugDraw.h>
#endif // INTERNAL_BUILD
#include <DetourCommon.h>

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

static BuildContext *m_ctx;
static rcConfig m_cfg;
static rcHeightfield *m_solid;
static unsigned char *m_triareas;
static rcCompactHeightfield *m_chf;
static rcContourSet *m_cset;
static rcPolyMesh *m_pmesh;
static rcPolyMeshDetail* m_dmesh;

static dtNavMesh* m_navMesh;
static dtNavMeshQuery* m_navQuery;

enum SamplePartitionType
{
    SAMPLE_PARTITION_WATERSHED,
    SAMPLE_PARTITION_MONOTONE,
    SAMPLE_PARTITION_LAYERS
};

bool CreateRecastNavMesh(game_state *GameState)
{
    LogMessage("Building Recast NavMesh");

    BuildContext ctx;
    m_ctx = &ctx;

    std::vector<int> LevelColliderTriangles;
    int LoadingLevelColliderPointsIterator = 0;
    for (u32 ColliderIndex = 0; ColliderIndex < GameState->LoadingLevelColliderSpans.lenu(); ++ColliderIndex)
    {
        u32 Span = GameState->LoadingLevelColliderSpans[ColliderIndex];
        vec3 *PointCloudPtr = &GameState->LoadingLevelColliderPoints[LoadingLevelColliderPointsIterator];

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
    for (u32 i = 0; i < GameState->LoadingLevelColliderPoints.lenu(); ++i)
    {
        vec3 point = GameState->LoadingLevelColliderPoints[i];
        min.x = GM_min(min.x, point.x);
        min.y = GM_min(min.y, point.y);
        min.z = GM_min(min.z, point.z);
        max.x = GM_max(max.x, point.x);
        max.y = GM_max(max.y, point.y);
        max.z = GM_max(max.z, point.z);
    }

    const float* bmin = (float*)&min;
    const float* bmax = (float*)&max;
    const float* verts = (float*)GameState->LoadingLevelColliderPoints.data;
    const int nverts = (int)GameState->LoadingLevelColliderPoints.lenu()*3;
    const int* tris = (int*)LevelColliderTriangles.data();
    const int ntris = (int)LevelColliderTriangles.size() / 3;

    // NOTE(Kevin): Need to tweak these configs to fit my levels
    //              5 cell size and 2 cell height seems to work ok
    
    float AgentHeight = 64.0f;
    float AgentRadius = 8.0f;
    float AgentMaxClimb = 5.f;

    //
    // Step 1. Initialize build config.
    //
    
    // Init build configuration from GUI
    memset(&m_cfg, 0, sizeof(m_cfg));
    /// The xz-plane cell size to use for fields. [Limit: > 0] [Units: wu] 
    m_cfg.cs = 4.0f;//8;
    /// The y-axis cell size to use for fields. [Limit: > 0] [Units: wu]
    m_cfg.ch = 1.2f;//6;
    /// The maximum slope that is considered walkable. [Limits: 0 <= value < 90] [Units: Degrees] 
    m_cfg.walkableSlopeAngle = 45;
    /// Minimum floor to 'ceiling' height that will still allow the floor area to 
    /// be considered walkable. [Limit: >= 3] [Units: vx] 
    m_cfg.walkableHeight = (int)ceilf(AgentHeight / m_cfg.ch);
    /// Maximum ledge height that is considered to still be traversable. [Limit: >=0] [Units: vx]
    // should be around 4...
    m_cfg.walkableClimb = (int)floorf(AgentMaxClimb / m_cfg.ch);
    /// The distance to erode/shrink the walkable area of the heightfield away from 
    /// obstructions.  [Limit: >=0] [Units: vx] 
    m_cfg.walkableRadius = (int)ceilf(AgentRadius / m_cfg.cs);
    /// The maximum allowed length for contour edges along the border of the mesh. [Limit: >=0] [Units: vx] 
    /// maxEdgeLength to zero will disabled the edge length feature. See rcBuildContours.
    m_cfg.maxEdgeLen = 0; //1000;//40;
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
    m_cfg.detailSampleMaxError = 0.1f;//m_cfg.ch * 1.f;//m_cellHeight * m_detailSampleMaxError;
    
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

#if INTERNAL_BUILD
    const bool m_keepInterResults = true;
#else
    const bool m_keepInterResults = false;
#endif // INTERNAL_BUILD
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

    //
    // Step 8. Create Detour data from Recast poly mesh.
    //

    // Only build the detour navmesh if we do not exceed the limit.
    if (m_cfg.maxVertsPerPoly <= DT_VERTS_PER_POLYGON)
    {
        unsigned char* navData = 0;
        int navDataSize = 0;

        // NOTE(Kevin): Polygon flags MUST be set to non-zero otherwise 
        // the query filter will reject it.
        // Update poly flags from areas.
        for (int i = 0; i < m_pmesh->npolys; ++i)
        {
            m_pmesh->flags[i] = 0x01;

            // u8 areas = m_pmesh->areas[i];
            // u16 flags = m_pmesh->flags[i];

            // if (m_pmesh->areas[i] == RC_WALKABLE_AREA)
            //     m_pmesh->areas[i] = SAMPLE_POLYAREA_GROUND;
                
            // if (m_pmesh->areas[i] == SAMPLE_POLYAREA_GROUND ||
            //     m_pmesh->areas[i] == SAMPLE_POLYAREA_GRASS ||
            //     m_pmesh->areas[i] == SAMPLE_POLYAREA_ROAD)
            // {
            //     m_pmesh->flags[i] = SAMPLE_POLYFLAGS_WALK;
            // }
            // else if (m_pmesh->areas[i] == SAMPLE_POLYAREA_WATER)
            // {
            //     m_pmesh->flags[i] = SAMPLE_POLYFLAGS_SWIM;
            // }
            // else if (m_pmesh->areas[i] == SAMPLE_POLYAREA_DOOR)
            // {
            //     m_pmesh->flags[i] = SAMPLE_POLYFLAGS_WALK | SAMPLE_POLYFLAGS_DOOR;
            // }
        }


        dtNavMeshCreateParams params;
        memset(&params, 0, sizeof(params));
        params.verts = m_pmesh->verts;
        params.vertCount = m_pmesh->nverts;
        params.polys = m_pmesh->polys;
        params.polyAreas = m_pmesh->areas;
        params.polyFlags = m_pmesh->flags;
        params.polyCount = m_pmesh->npolys;
        params.nvp = m_pmesh->nvp;
        params.detailMeshes = m_dmesh->meshes;
        params.detailVerts = m_dmesh->verts;
        params.detailVertsCount = m_dmesh->nverts;
        params.detailTris = m_dmesh->tris;
        params.detailTriCount = m_dmesh->ntris;
        // params.offMeshConVerts = m_geom->getOffMeshConnectionVerts();
        // params.offMeshConRad = m_geom->getOffMeshConnectionRads();
        // params.offMeshConDir = m_geom->getOffMeshConnectionDirs();
        // params.offMeshConAreas = m_geom->getOffMeshConnectionAreas();
        // params.offMeshConFlags = m_geom->getOffMeshConnectionFlags();
        // params.offMeshConUserID = m_geom->getOffMeshConnectionId();
        // params.offMeshConCount = m_geom->getOffMeshConnectionCount();
        params.offMeshConVerts = NULL;
        params.offMeshConRad = NULL;
        params.offMeshConDir = NULL;
        params.offMeshConAreas = NULL;
        params.offMeshConFlags = NULL;
        params.offMeshConUserID = NULL;
        params.offMeshConCount = NULL;
        params.walkableHeight = AgentHeight;
        params.walkableRadius = AgentRadius;
        params.walkableClimb = AgentMaxClimb;
        rcVcopy(params.bmin, m_pmesh->bmin);
        rcVcopy(params.bmax, m_pmesh->bmax);
        params.cs = m_cfg.cs;
        params.ch = m_cfg.ch;
        params.buildBvTree = true;
        
        if (!dtCreateNavMeshData(&params, &navData, &navDataSize))
        {
            m_ctx->log(RC_LOG_ERROR, "Could not build Detour navmesh.");
            return false;
        }
        
        m_navQuery = dtAllocNavMeshQuery();

        m_navMesh = dtAllocNavMesh();
        if (!m_navMesh)
        {
            dtFree(navData);
            m_ctx->log(RC_LOG_ERROR, "Could not create Detour navmesh");
            return false;
        }
        
        dtStatus status;
        
        status = m_navMesh->init(navData, navDataSize, DT_TILE_FREE_DATA);
        if (dtStatusFailed(status))
        {
            dtFree(navData);
            m_ctx->log(RC_LOG_ERROR, "Could not init Detour navmesh");
            return false;
        }
        
        status = m_navQuery->init(m_navMesh, 2048);
        if (dtStatusFailed(status))
        {
            m_ctx->log(RC_LOG_ERROR, "Could not init Detour navmesh query");
            return false;
        }
    }

    return true;
}

void DestroyRecastNavMesh()
{
    delete [] m_triareas;
    m_triareas = 0;
    rcFreeHeightField(m_solid);
    m_solid = 0;
    rcFreeCompactHeightfield(m_chf);
    m_chf = 0;
    rcFreeContourSet(m_cset);
    m_cset = 0;
    rcFreePolyMesh(m_pmesh);
    m_pmesh = 0;
    rcFreePolyMeshDetail(m_dmesh);
    m_dmesh = 0;
    dtFreeNavMesh(m_navMesh);
    m_navMesh = 0;

    dtFreeNavMeshQuery(m_navQuery);
    dtFreeNavMesh(m_navMesh);
}

static int dtMergeCorridorStartMoved(dtPolyRef* path, const int npath, const int maxPath,
                              const dtPolyRef* visited, const int nvisited)
{
    int furthestPath = -1;
    int furthestVisited = -1;
    
    // Find furthest common polygon.
    for (int i = npath-1; i >= 0; --i)
    {
        bool found = false;
        for (int j = nvisited-1; j >= 0; --j)
        {
            if (path[i] == visited[j])
            {
                furthestPath = i;
                furthestVisited = j;
                found = true;
            }
        }
        if (found)
            break;
    }
    
    // If no intersection found just return current path. 
    if (furthestPath == -1 || furthestVisited == -1)
        return npath;
    
    // Concatenate paths.   
    
    // Adjust beginning of the buffer to include the visited.
    const int req = nvisited - furthestVisited;
    const int orig = dtMin(furthestPath+1, npath);
    int size = dtMax(0, npath-orig);
    if (req+size > maxPath)
        size = maxPath-req;
    if (size > 0)
        memmove(path+req, path+orig, size*sizeof(dtPolyRef));

    // Store visited
    for (int i = 0, n = dtMin(req, maxPath); i < n; ++i)
        path[i] = visited[(nvisited-1)-i];

    return req+size;
}

// This function checks if the path has a small U-turn, that is,
// a polygon further in the path is adjacent to the first polygon
// in the path. If that happens, a shortcut is taken.
// This can happen if the target (T) location is at tile boundary,
// and we're (S) approaching it parallel to the tile edge.
// The choice at the vertex can be arbitrary, 
//  +---+---+
//  |:::|:::|
//  +-S-+-T-+
//  |:::|   | <-- the step can end up in here, resulting U-turn path.
//  +---+---+
static int fixupShortcuts(dtPolyRef* path, int npath, dtNavMeshQuery* navQuery)
{
    if (npath < 3)
        return npath;

    // Get connected polygons
    static const int maxNeis = 16;
    dtPolyRef neis[maxNeis];
    int nneis = 0;

    const dtMeshTile* tile = 0;
    const dtPoly* poly = 0;
    if (dtStatusFailed(navQuery->getAttachedNavMesh()->getTileAndPolyByRef(path[0], &tile, &poly)))
        return npath;
    
    for (unsigned int k = poly->firstLink; k != DT_NULL_LINK; k = tile->links[k].next)
    {
        const dtLink* link = &tile->links[k];
        if (link->ref != 0)
        {
            if (nneis < maxNeis)
                neis[nneis++] = link->ref;
        }
    }

    // If any of the neighbour polygons is within the next few polygons
    // in the path, short cut to that polygon directly.
    static const int maxLookAhead = 6;
    int cut = 0;
    for (int i = dtMin(maxLookAhead, npath) - 1; i > 1 && cut == 0; i--) {
        for (int j = 0; j < nneis; j++)
        {
            if (path[i] == neis[j]) {
                cut = i;
                break;
            }
        }
    }
    if (cut > 1)
    {
        int offset = cut-1;
        npath -= offset;
        for (int i = 1; i < npath; i++)
            path[i] = path[i+offset];
    }

    return npath;
}

inline bool inRange(const float* v1, const float* v2, const float r, const float h)
{
    const float dx = v2[0] - v1[0];
    const float dy = v2[1] - v1[1];
    const float dz = v2[2] - v1[2];
    return (dx*dx + dz*dz) < r*r && fabsf(dy) < h;
}

static bool getSteerTarget(dtNavMeshQuery* navQuery, const float* startPos, const float* endPos,
                           const float minTargetDist,
                           const dtPolyRef* path, const int pathSize,
                           float* steerPos, unsigned char& steerPosFlag, dtPolyRef& steerPosRef,
                           float* outPoints = 0, int* outPointCount = 0)                             
{
    // Find steer target.
    static const int MAX_STEER_POINTS = 3;
    float steerPath[MAX_STEER_POINTS*3];
    unsigned char steerPathFlags[MAX_STEER_POINTS];
    dtPolyRef steerPathPolys[MAX_STEER_POINTS];
    int nsteerPath = 0;
    navQuery->findStraightPath(startPos, endPos, path, pathSize,
                               steerPath, steerPathFlags, steerPathPolys, &nsteerPath, MAX_STEER_POINTS);
    if (!nsteerPath)
        return false;
        
    if (outPoints && outPointCount)
    {
        *outPointCount = nsteerPath;
        for (int i = 0; i < nsteerPath; ++i)
            dtVcopy(&outPoints[i*3], &steerPath[i*3]);
    }

    
    // Find vertex far enough to steer to.
    int ns = 1;
    while (ns < nsteerPath)
    {
        // Stop at Off-Mesh link or when point is further than slop away.
        // (steerPathFlags[ns] & DT_STRAIGHTPATH_OFFMESH_CONNECTION) ||
        if (!inRange(&steerPath[ns*3], startPos, minTargetDist, 1000.0f))
            break;
        ns++;
    }
    // Failed to find good point to steer to.
    if (ns >= nsteerPath)
        return false;
    
    dtVcopy(steerPos, &steerPath[ns*3]);
    steerPos[1] = startPos[1];
    steerPosFlag = steerPathFlags[ns];
    steerPosRef = steerPathPolys[ns];
    
    return true;
}

static const int MAX_POLYS = 256;
static dtPolyRef m_polys[MAX_POLYS];
static int m_npolys;
static float m_smoothPath[MAX_SMOOTH*3];
static int m_nsmoothPath;
static dtPolyRef m_startRef;
static dtPolyRef m_endRef;
static float m_spos[3];
static float m_epos[3];
static dtQueryFilter m_filter;

void TOOLMODE_PATHFIND_FOLLOW()
{

// #if INTERNAL_BUILD
//     printf("pi  %f %f %f  %f %f %f  0x%x 0x%x\n",
//            m_spos[0],m_spos[1],m_spos[2], m_epos[0],m_epos[1],m_epos[2],
//            m_filter.getIncludeFlags(), m_filter.getExcludeFlags()); 
// #endif

    m_npolys = 0;

    m_navQuery->findPath(m_startRef, m_endRef, m_spos, m_epos, &m_filter, m_polys, &m_npolys, MAX_POLYS);

    m_nsmoothPath = 0;

    if (m_npolys)
    {
        // Iterate over the path to find smooth path on the detail mesh surface.
        dtPolyRef polys[MAX_POLYS];
        memcpy(polys, m_polys, sizeof(dtPolyRef)*m_npolys); 
        int npolys = m_npolys;
        
        float iterPos[3], targetPos[3];
        m_navQuery->closestPointOnPoly(m_startRef, m_spos, iterPos, 0);
        m_navQuery->closestPointOnPoly(polys[npolys-1], m_epos, targetPos, 0);
        
        const float STEP_SIZE = 64.f;
        const float SLOP = 2.f;
        
        m_nsmoothPath = 0;
        
        dtVcopy(&m_smoothPath[m_nsmoothPath*3], iterPos);
        m_nsmoothPath++;
        
        // Move towards target a small advancement at a time until target reached or
        // when ran out of memory to store the path.
        while (npolys && m_nsmoothPath < MAX_SMOOTH)
        {
            // Find location to steer towards.
            float steerPos[3];
            unsigned char steerPosFlag;
            dtPolyRef steerPosRef;
            
            if (!getSteerTarget(m_navQuery, iterPos, targetPos, SLOP,
                                polys, npolys, steerPos, steerPosFlag, steerPosRef))
                break;
            
            bool endOfPath = (steerPosFlag & DT_STRAIGHTPATH_END) ? true : false;
            bool offMeshConnection = (steerPosFlag & DT_STRAIGHTPATH_OFFMESH_CONNECTION) ? true : false;
            
            // Find movement delta.
            float delta[3], len;
            dtVsub(delta, steerPos, iterPos);
            len = dtMathSqrtf(dtVdot(delta, delta));
            // If the steer target is end of path or off-mesh link, do not move past the location.
            if ((endOfPath || offMeshConnection) && len < STEP_SIZE)
                len = 1;
            else
                len = STEP_SIZE / len;
            float moveTgt[3];
            dtVmad(moveTgt, iterPos, delta, len);
            
            // Move
            float result[3];
            dtPolyRef visited[16];
            int nvisited = 0;
            m_navQuery->moveAlongSurface(polys[0], iterPos, moveTgt, &m_filter,
                                         result, visited, &nvisited, 16);

            // NOTE(Kevin): smooth path doesn't work across multiple polygons without
            //              the following function from DetourCrowd
            npolys = dtMergeCorridorStartMoved(polys, npolys, MAX_POLYS, visited, nvisited);
            npolys = fixupShortcuts(polys, npolys, m_navQuery);

            float h = 0;
            m_navQuery->getPolyHeight(polys[0], result, &h);
            result[1] = h;
            dtVcopy(iterPos, result);

            // Handle end of path and off-mesh links when close enough.
            if (endOfPath && inRange(iterPos, steerPos, SLOP, 1.0f))
            {
                // Reached end of path.
                dtVcopy(iterPos, targetPos);
                if (m_nsmoothPath < MAX_SMOOTH)
                {
                    dtVcopy(&m_smoothPath[m_nsmoothPath*3], iterPos);
                    m_nsmoothPath++;
                }
                break;
            }
            else if (offMeshConnection && inRange(iterPos, steerPos, SLOP, 1.0f))
            {
                // Reached off-mesh connection.
                float startPos[3], endPos[3];
                
                // Advance the path up to and over the off-mesh connection.
                dtPolyRef prevRef = 0, polyRef = polys[0];
                int npos = 0;
                while (npos < npolys && polyRef != steerPosRef)
                {
                    prevRef = polyRef;
                    polyRef = polys[npos];
                    npos++;
                }
                for (int i = npos; i < npolys; ++i)
                    polys[i-npos] = polys[i];
                npolys -= npos;
                
                // Handle the connection.
                dtStatus status = m_navMesh->getOffMeshConnectionPolyEndPoints(prevRef, polyRef, startPos, endPos);
                if (dtStatusSucceed(status))
                {
                    if (m_nsmoothPath < MAX_SMOOTH)
                    {
                        dtVcopy(&m_smoothPath[m_nsmoothPath*3], startPos);
                        m_nsmoothPath++;
                        // Hack to make the dotted path not visible during off-mesh connection.
                        if (m_nsmoothPath & 1)
                        {
                            dtVcopy(&m_smoothPath[m_nsmoothPath*3], startPos);
                            m_nsmoothPath++;
                        }
                    }
                    // Move position at the other side of the off-mesh link.
                    dtVcopy(iterPos, endPos);
                    float eh = 0.0f;
                    m_navQuery->getPolyHeight(polys[0], iterPos, &eh);
                    iterPos[1] = eh;
                }
            }
            
            // Store results.
            if (m_nsmoothPath < MAX_SMOOTH)
            {
                dtVcopy(&m_smoothPath[m_nsmoothPath*3], iterPos);
                m_nsmoothPath++;
            }
        }
    }

    // at this point m_smoothPath is populated with m_nsmoothPath vector3s
}

bool FindSmoothPathTo(vec3 Origin, vec3 Target, float *SmoothPath, int *SmoothPathCount)
{
    vec3 SearchHalfExtents = vec3(16.f,16.f,16.f);

    dtStatus Status = m_navQuery->findNearestPoly(
        (float*)&Origin, (float*)&SearchHalfExtents, &m_filter, 
        &m_startRef, m_spos);
    if (!dtStatusSucceed(Status))
        return false;

    Status = m_navQuery->findNearestPoly(
        (float*)&Target, (float*)&SearchHalfExtents, &m_filter, 
        &m_endRef, m_epos);
    if (!dtStatusSucceed(Status))
        return false;

    if (m_startRef == 0 || m_endRef == 0)
        return false;

    TOOLMODE_PATHFIND_FOLLOW();

    if (m_nsmoothPath > 0)
    {
        memcpy(SmoothPath, m_smoothPath, m_nsmoothPath * 3 * sizeof(float));
        *SmoothPathCount = m_nsmoothPath;
        return true;
    }
    else
    {
        return false;
    }
}

float nav_frand01()
{
    return RNG.frand01();
}

void GetRandomPointOnNavMesh(float *Point)
{
    dtPolyRef RandomPoly;
    dtStatus Status = m_navQuery->findRandomPoint(&m_filter, nav_frand01, 
        &RandomPoly, Point);
    ASSERT(dtStatusSucceed(Status));
}


#if INTERNAL_BUILD

// void DetourTesting()
// {
//     static dynamic_array<dtPolyRef> PolygonCorridor;
//     PolygonCorridor.setlen(256);
//     static int PathCount;

//     static vec3 StraightPathPoints[16];
//     static u8 StraightPathFlags[16];
//     static dtPolyRef StraightPathPolyRefs[16];
//     static int StraightPathPointCount = 0;

//     static int Iter = 1;

//     if (KeysPressed[SDL_SCANCODE_N])
//     {
//         // runonce = false;
//         LogMessage("Pathing towards player!");
//         Iter = 1;

//         vec3 EnemyStartPos = EnemyPosition;//vec3(4.f, 4.f, 4.f);
//         vec3 SearchHalfExtents = vec3(16.f,16.f,16.f);
//         // dtQueryFilter QueryFilter;

//         dtStatus Status = m_navQuery->findNearestPoly(
//             (float*)&EnemyStartPos, (float*)&SearchHalfExtents, &m_filter, 
//             &m_startRef, m_spos);
//         //dtStatus Status = m_navQuery->findRandomPoint(&m_filter, frand01, 
//         //    &StartNearestPoly, (float*)&StartNearestPoint);
//         ASSERT(dtStatusSucceed(Status));

//         // EnemyPosition = StartNearestPoint;

//         Status = m_navQuery->findNearestPoly(
//             (float*)&g_GameState.Player.Root, (float*)&SearchHalfExtents, &m_filter, 
//             &m_endRef, m_epos);
//         ASSERT(dtStatusSucceed(Status));

//         TOOLMODE_PATHFIND_FOLLOW();

//         // Status = m_navQuery->findPath(StartNearestPoly, EndNearestPoly, 
//         //     (float*)&StartNearestPoint, (float*)&EndNearestPoint, &m_filter,
//         //     PolygonCorridor.data, &PathCount, 256);
//         // ASSERT(dtStatusSucceed(Status));

//         // m_navQuery->findStraightPath(
//         //     (float*)&StartNearestPoint, (float*)&EndNearestPoint, PolygonCorridor.data, PathCount,
//         //     (float*)StraightPathPoints, StraightPathFlags, StraightPathPolyRefs, &StraightPathPointCount,
//         //     16); // DT_STRAIGHTPATH_AREA_CROSSINGS | DT_STRAIGHTPATH_ALL_CROSSINGS
//     }

//     // NOTE(Kevin): I think using the smooth path finding might be slow. Navmesh doesn't really care about Y coord,
//     //              so maybe I should just use physics or something to correctly position the enemy above ground.

//     if (Iter < m_nsmoothPath)
//     {
//         vec3 SteerPoint = *(vec3*)&m_smoothPath[Iter*3];
//         vec3 DirToSteerPoint = Normalize(SteerPoint - EnemyPosition);
//         vec3 EnemyMoveDelta = DirToSteerPoint * 64.f * DeltaTime;
//         float DistTravelled = Magnitude(EnemyMoveDelta);
//         float DistToSteerPoint = Magnitude(SteerPoint - EnemyPosition);
//         if (DistTravelled >= DistToSteerPoint)
//         {
//             EnemyPosition = SteerPoint;
//             ++Iter;
//         }
//         else
//         {
//             EnemyPosition += EnemyMoveDelta;
//         }
//     }
// }

void DebugDrawRecast(duDebugDraw *DebugDrawer, recast_debug_drawmode DrawMode)
{
    if (m_navMesh && m_navQuery &&
        (DrawMode == DRAWMODE_NAVMESH ||
        DrawMode == DRAWMODE_NAVMESH_TRANS ||
        DrawMode == DRAWMODE_NAVMESH_BVTREE ||
        DrawMode == DRAWMODE_NAVMESH_NODES ||
        DrawMode == DRAWMODE_NAVMESH_INVIS))
    {
        if (DrawMode != DRAWMODE_NAVMESH_INVIS)
            duDebugDrawNavMeshWithClosedList(DebugDrawer, *m_navMesh, *m_navQuery, 0);
        if (DrawMode == DRAWMODE_NAVMESH_BVTREE)
            duDebugDrawNavMeshBVTree(DebugDrawer, *m_navMesh);
        if (DrawMode == DRAWMODE_NAVMESH_NODES)
            duDebugDrawNavMeshNodes(DebugDrawer, *m_navQuery);
        // duDebugDrawNavMeshPolysWithFlags(DebugDrawer, *m_navMesh, SAMPLE_POLYFLAGS_DISABLED, duRGBA(0,0,0,128));
    }
        
    glDepthMask(GL_TRUE);

    if (m_chf && DrawMode == DRAWMODE_COMPACT)
        duDebugDrawCompactHeightfieldSolid(DebugDrawer, *m_chf);

    if (m_chf && DrawMode == DRAWMODE_COMPACT_DISTANCE)
        duDebugDrawCompactHeightfieldDistance(DebugDrawer, *m_chf);
    if (m_chf && DrawMode == DRAWMODE_COMPACT_REGIONS)
        duDebugDrawCompactHeightfieldRegions(DebugDrawer, *m_chf);
    if (m_solid && DrawMode == DRAWMODE_VOXELS)
    {
        // glEnable(GL_FOG);
        duDebugDrawHeightfieldSolid(DebugDrawer, *m_solid);
        // glDisable(GL_FOG);
    }
    if (m_solid && DrawMode == DRAWMODE_VOXELS_WALKABLE)
    {
        // glEnable(GL_FOG);
        duDebugDrawHeightfieldWalkable(DebugDrawer, *m_solid);
        // glDisable(GL_FOG);
    }
    if (m_cset && DrawMode == DRAWMODE_RAW_CONTOURS)
    {
        glDepthMask(GL_FALSE);
        duDebugDrawRawContours(DebugDrawer, *m_cset);
        glDepthMask(GL_TRUE);
    }
    if (m_cset && DrawMode == DRAWMODE_BOTH_CONTOURS)
    {
        // glDepthMask(GL_FALSE);
        duDebugDrawRawContours(DebugDrawer, *m_cset, 0.5f);
        duDebugDrawContours(DebugDrawer, *m_cset);
        // glDepthMask(GL_TRUE);
    }
    if (m_cset && DrawMode == DRAWMODE_CONTOURS)
    {
        // glDepthMask(GL_FALSE);
        duDebugDrawContours(DebugDrawer, *m_cset);
        // glDepthMask(GL_TRUE);
    }
    if (m_chf && m_cset && DrawMode == DRAWMODE_REGION_CONNECTIONS)
    {
        duDebugDrawCompactHeightfieldRegions(DebugDrawer, *m_chf);
            
        // glDepthMask(GL_FALSE);
        duDebugDrawRegionConnections(DebugDrawer, *m_cset);
        // glDepthMask(GL_TRUE);
    }
    if (m_pmesh && DrawMode == DRAWMODE_POLYMESH)
    {
        // glDepthMask(GL_FALSE);
        duDebugDrawPolyMesh(DebugDrawer, *m_pmesh);
        // glDepthMask(GL_TRUE);
    }
    if (m_dmesh && DrawMode == DRAWMODE_POLYMESH_DETAIL)
    {
        // glDepthMask(GL_FALSE);
        duDebugDrawPolyMeshDetail(DebugDrawer, *m_dmesh);
        // glDepthMask(GL_TRUE);
    }

    // GLHasErrors();
}

// static void DebugDrawAgent(const float* pos, float r, float h, float c, const unsigned int col)
// {
//     duDebugDraw& dd = RecastDebugDrawer;
    
//     dd.depthMask(false);
    
//     // Agent dimensions.    
//     duDebugDrawCylinderWire(&dd, pos[0]-r, pos[1]+0.02f, pos[2]-r, pos[0]+r, pos[1]+h, pos[2]+r, col, 2.0f);

//     duDebugDrawCircle(&dd, pos[0],pos[1]+c,pos[2],r,duRGBA(0,0,0,64),1.0f);

//     unsigned int colb = duRGBA(0,0,0,196);
//     dd.begin(DU_DRAW_LINES);
//     dd.vertex(pos[0], pos[1]-c, pos[2], colb);
//     dd.vertex(pos[0], pos[1]+c, pos[2], colb);
//     dd.vertex(pos[0]-r/2, pos[1]+0.02f, pos[2], colb);
//     dd.vertex(pos[0]+r/2, pos[1]+0.02f, pos[2], colb);
//     dd.vertex(pos[0], pos[1]+0.02f, pos[2]-r/2, colb);
//     dd.vertex(pos[0], pos[1]+0.02f, pos[2]+r/2, colb);
//     dd.end();
    
//     dd.depthMask(true);
// }

void DebugDrawFollowPath(support_renderer_t *SupportRenderer)
{
    // duDebugDraw& dd = RecastDebugDrawer;

    // const unsigned int startCol = duRGBA(128,25,0,192);
    // const unsigned int endCol = duRGBA(51,102,0,129);
    // const unsigned int pathCol = duRGBA(0,0,0,64);
    
    // NOTE(Kevin) 2025-03-15: I don't wanna see the agent or the start end nav mesh poly debug drawn
    // const float agentHeight = 8.0f;
    // const float agentRadius = 8.0f;
    // const float agentClimb = 5.f;
    // dd.depthMask(false);
    // DebugDrawAgent(m_spos, agentRadius, agentHeight, agentClimb, startCol);
    // DebugDrawAgent(m_epos, agentRadius, agentHeight, agentClimb, endCol);
    // dd.depthMask(true);

    // duDebugDrawNavMeshPoly(&dd, *m_navMesh, m_startRef, startCol);
    // duDebugDrawNavMeshPoly(&dd, *m_navMesh, m_endRef, endCol);
    
    // if (m_npolys)
    // {
    //     for (int i = 0; i < m_npolys; ++i)
    //     {
    //         if (m_polys[i] == m_startRef || m_polys[i] == m_endRef)
    //             continue;
    //         duDebugDrawNavMeshPoly(&dd, *m_navMesh, m_polys[i], pathCol);
    //     }
    // }

    // if (m_nsmoothPath)
    // {
    //     dd.depthMask(false);
    //     const unsigned int spathCol = duRGBA(0,0,0,220);
    //     dd.begin(DU_DRAW_LINES, 3.0f);
    //     for (int i = 0; i < m_nsmoothPath; ++i)
    //         dd.vertex(m_smoothPath[i*3], m_smoothPath[i*3+1]+0.1f, m_smoothPath[i*3+2], spathCol);
    //     dd.end();
    //     dd.depthMask(true);
    // }

    // Just using support renderer primitives
    if (m_nsmoothPath)
    {
        for (int i = 1; i < m_nsmoothPath; ++i)
        {
            int a = i-1;
            int b = i;
            float ax = m_smoothPath[a*3];
            float ay = m_smoothPath[a*3+1]+0.1f;
            float az = m_smoothPath[a*3+2];
            float bx = m_smoothPath[b*3];
            float by = m_smoothPath[b*3+1]+0.1f;
            float bz = m_smoothPath[b*3+2];
            SupportRenderer->DrawLine(vec3(ax,ay,az), vec3(bx,by,bz), vec4(0,0,0,0.85f), 2.f);
        }
    }
}

void recast_debug_draw_gl3_t::Ready()
{

}

void recast_debug_draw_gl3_t::depthMask(bool state)
{
    // if (state)
    //     glEnable(GL_DEPTH_TEST);
    // else
    //     glDisable(GL_DEPTH_TEST);
}

void recast_debug_draw_gl3_t::texture(bool state)
{
    // will not implement
}

void recast_debug_draw_gl3_t::begin(duDebugDrawPrimitives prim, float size)
{
    CurrentPrimitiveDrawMode = prim;
    CurrentSize = size;
    VertexBuffer.setlen(0);
}

void recast_debug_draw_gl3_t::vertex(const float* pos, unsigned int color)
{
    u8* Colors = (u8*)&color;
    float r = float(Colors[0]) / 255.f;
    float g = float(Colors[1]) / 255.f;
    float b = float(Colors[2]) / 255.f;
    float a = float(Colors[3]) / 255.f;

    if (CurrentPrimitiveDrawMode == DU_DRAW_TRIS)
    {
        if (SupportRenderer->PRIMITIVE_TRIS_VB.lenu() + 7 > SupportRenderer->PRIMITIVE_TRIS_VB.cap())
        {
            LogError("PRIMITIVE_TRIS_VB at capacity: cannot insert debug tris.");
            return;
        }

        SupportRenderer->PRIMITIVE_TRIS_VB.put(pos[0]);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(pos[1]);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(pos[2]);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(r);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(g);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(b);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(a);
    }
    else if (CurrentPrimitiveDrawMode == DU_DRAW_LINES)
    {
        if (SupportRenderer->PRIMITIVE_LINES_VB.lenu() + 7 > SupportRenderer->PRIMITIVE_LINES_VB.cap())
        {
            LogError("PRIMITIVE_LINES_VB at capacity: cannot insert debug lines.");
            return;
        }

        SupportRenderer->PRIMITIVE_LINES_VB.put(pos[0]);
        SupportRenderer->PRIMITIVE_LINES_VB.put(pos[1]);
        SupportRenderer->PRIMITIVE_LINES_VB.put(pos[2]);
        SupportRenderer->PRIMITIVE_LINES_VB.put(r);
        SupportRenderer->PRIMITIVE_LINES_VB.put(g);
        SupportRenderer->PRIMITIVE_LINES_VB.put(b);
        SupportRenderer->PRIMITIVE_LINES_VB.put(a);
    }
    else if (CurrentPrimitiveDrawMode == DU_DRAW_POINTS)
    {
        SupportRenderer->DrawSolidRect(
            vec3(pos[0], pos[1], pos[2]),
            -GameState->Player.PlayerCam.Direction,
            CurrentSize,
            vec4(r,g,b,a));
    }
    else if (CurrentPrimitiveDrawMode == DU_DRAW_QUADS)
    {
        // Quad processed later
        VertexBuffer.put(pos[0]);
        VertexBuffer.put(pos[1]);
        VertexBuffer.put(pos[2]);
        VertexBuffer.put(r);
        VertexBuffer.put(g);
        VertexBuffer.put(b);
        VertexBuffer.put(a);
    }
}

void recast_debug_draw_gl3_t::vertex(const float x, const float y, const float z, unsigned int color)
{
    u8* Colors = (u8*)&color;
    float r = float(Colors[0]) / 255.f;
    float g = float(Colors[1]) / 255.f;
    float b = float(Colors[2]) / 255.f;
    float a = float(Colors[3]) / 255.f;

    if (CurrentPrimitiveDrawMode == DU_DRAW_TRIS)
    {
        if (SupportRenderer->PRIMITIVE_TRIS_VB.lenu() + 7 > SupportRenderer->PRIMITIVE_TRIS_VB.cap())
        {
            LogError("PRIMITIVE_TRIS_VB at capacity: cannot insert debug tris.");
            return;
        }

        SupportRenderer->PRIMITIVE_TRIS_VB.put(x);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(y);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(z);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(r);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(g);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(b);
        SupportRenderer->PRIMITIVE_TRIS_VB.put(a);
    }
    else if (CurrentPrimitiveDrawMode == DU_DRAW_LINES)
    {
        if (SupportRenderer->PRIMITIVE_LINES_VB.lenu() + 7 > SupportRenderer->PRIMITIVE_LINES_VB.cap())
        {
            LogError("PRIMITIVE_LINES_VB at capacity: cannot insert debug lines.");
            return;
        }

        SupportRenderer->PRIMITIVE_LINES_VB.put(x);
        SupportRenderer->PRIMITIVE_LINES_VB.put(y);
        SupportRenderer->PRIMITIVE_LINES_VB.put(z);
        SupportRenderer->PRIMITIVE_LINES_VB.put(r);
        SupportRenderer->PRIMITIVE_LINES_VB.put(g);
        SupportRenderer->PRIMITIVE_LINES_VB.put(b);
        SupportRenderer->PRIMITIVE_LINES_VB.put(a);
    }
    else if (CurrentPrimitiveDrawMode == DU_DRAW_POINTS)
    {
        SupportRenderer->DrawSolidRect(
            vec3(x,y,z),
            -GameState->Player.PlayerCam.Direction,
            CurrentSize,
            vec4(r,g,b,a));
    }
    else if (CurrentPrimitiveDrawMode == DU_DRAW_QUADS)
    {
        // Quad processed later
        VertexBuffer.put(x);
        VertexBuffer.put(y);
        VertexBuffer.put(z);
        VertexBuffer.put(r);
        VertexBuffer.put(g);
        VertexBuffer.put(b);
        VertexBuffer.put(a);
    }
}

void recast_debug_draw_gl3_t::vertex(const float* pos, unsigned int color, const float* uv)
{
    vertex(pos[0],pos[1],pos[2],color);
    // fuck uv
}

void recast_debug_draw_gl3_t::vertex(const float x, const float y, const float z, unsigned int color, const float u, const float v)
{
    vertex(x,y,z,color);
    // fuck uv
}

void recast_debug_draw_gl3_t::end()
{
    if (CurrentPrimitiveDrawMode == GL_QUADS)
    {
        // GL_QUAD not supported on my pc so I need to change this quad vb to tris vb
        if (SupportRenderer->PRIMITIVE_TRIS_VB.lenu() + VertexBuffer.lenu() > SupportRenderer->PRIMITIVE_TRIS_VB.cap())
        {
            LogError("PRIMITIVE_TRIS_VB at capacity: cannot insert debug quads.");
            return;
        }
        float *copyptr = SupportRenderer->PRIMITIVE_TRIS_VB.addnptr(size_t(VertexBuffer.lenu() * 1.5f));
        size_t vbi = 0;
        for (size_t i = 0; i < VertexBuffer.lenu(); i += 36) // 4*9*sizeof(float)
        {
            // TL -> BL -> BR -> TR
            // i+0   i+7   i+14  i+21
            float *TL = &VertexBuffer[i+0];
            float *BL = &VertexBuffer[i+7];
            float *BR = &VertexBuffer[i+14];
            float *TR = &VertexBuffer[i+21];
            memcpy(&copyptr[vbi+0], TL, 7*sizeof(float));
            memcpy(&copyptr[vbi+7], BL, 7*sizeof(float));
            memcpy(&copyptr[vbi+14], BR, 7*sizeof(float));
            memcpy(&copyptr[vbi+21], TL, 7*sizeof(float));
            memcpy(&copyptr[vbi+28], BR, 7*sizeof(float));
            memcpy(&copyptr[vbi+35], TR, 7*sizeof(float));
            vbi += 42; // 6*7*sizeof(float)
        }
    }
    VertexBuffer.setlen(0);
}

#endif // INTERNAL_BUILD
