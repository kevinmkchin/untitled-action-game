#include <Recast.h>
#include <DebugDraw.h>
#include <RecastDebugDraw.h>
#include <DetourNavMesh.h>
#include <DetourNavMeshBuilder.h>
#include <DetourNavMeshQuery.h>

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

static dtNavMesh* m_navMesh;
static dtNavMeshQuery* m_navQuery;

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

void DetourTesting()
{
    static dynamic_array<dtPolyRef> PolygonCorridor;
    PolygonCorridor.setlen(256);
    static int PathCount;

    static vec3 StraightPathPoints[16];
    static u8 StraightPathFlags[16];
    static dtPolyRef StraightPathPolyRefs[16];
    static int StraightPathPointCount = 0;

    static int Iter = 1;

    // static bool runonce = true;
    // if (runonce)
    if (KeysPressed[SDL_SCANCODE_N])
    {
        // runonce = false;
        LogMessage("Pathing towards player!");
        Iter = 1;

        vec3 EnemyStartPos = EnemyPosition;//vec3(4.f, 4.f, 4.f);
        vec3 SearchHalfExtents = vec3(16.f,16.f,16.f);
        dtQueryFilter QueryFilter;

        dtPolyRef StartNearestPoly;
        vec3      StartNearestPoint;

        dtStatus Status = m_navQuery->findNearestPoly(
            (float*)&EnemyStartPos, (float*)&SearchHalfExtents, &QueryFilter, 
            &StartNearestPoly, (float*)&StartNearestPoint);
        //dtStatus Status = m_navQuery->findRandomPoint(&QueryFilter, frand, 
        //    &StartNearestPoly, (float*)&StartNearestPoint);
        ASSERT(dtStatusSucceed(Status));

        LogMessage("start nearest poly point %f, %f, %f", 
            StartNearestPoint.x, StartNearestPoint.y, StartNearestPoint.z);

        EnemyPosition = StartNearestPoint;

        dtPolyRef EndNearestPoly;
        vec3      EndNearestPoint;

        Status = m_navQuery->findNearestPoly(
            (float*)&Player.Root, (float*)&SearchHalfExtents, &QueryFilter, 
            &EndNearestPoly, (float*)&EndNearestPoint);
        ASSERT(dtStatusSucceed(Status));

        LogMessage("end nearest poly point %f, %f, %f", 
            EndNearestPoint.x, EndNearestPoint.y, EndNearestPoint.z);

        Status = m_navQuery->findPath(StartNearestPoly, EndNearestPoly, 
            (float*)&StartNearestPoint, (float*)&EndNearestPoint, &QueryFilter,
            PolygonCorridor.data, &PathCount, 256);
        ASSERT(dtStatusSucceed(Status));


        m_navQuery->findStraightPath(
            (float*)&StartNearestPoint, (float*)&EndNearestPoint, PolygonCorridor.data, PathCount,
            (float*)StraightPathPoints, StraightPathFlags, StraightPathPolyRefs, &StraightPathPointCount,
            16); // DT_STRAIGHTPATH_AREA_CROSSINGS | DT_STRAIGHTPATH_ALL_CROSSINGS
    }

    if (Iter < StraightPathPointCount)
    {
        vec3 SteerPoint = StraightPathPoints[Iter];
        vec3 DirToSteerPoint = Normalize(SteerPoint - EnemyPosition);
        vec3 EnemyMoveDelta = DirToSteerPoint * 64.f * DeltaTime;
        float DistTravelled = Magnitude(EnemyMoveDelta);
        float DistToSteerPoint = Magnitude(SteerPoint - EnemyPosition);
        if (DistTravelled >= DistToSteerPoint)
        {
            EnemyPosition = SteerPoint;
            ++Iter;
        }
        else
        {
            EnemyPosition += EnemyMoveDelta;
        }
    }
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
    
    vec3 ColorTable[] = {
        vec3(0.91f,0.59f,0.48f),
        vec3(1.00f,1.00f,0.00f),
        vec3(0.31f,0.58f,0.80f),
        vec3(1.00f,0.50f,0.00f),
        vec3(0.00f,1.00f,1.00f),
        vec3(0.58f,0.00f,0.83f),
        vec3(0.13f,0.55f,0.13f),
    };

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
        
        vec3 Color = ColorTable[i%7];

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
                VertexBuffer.push_back(Color);
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
