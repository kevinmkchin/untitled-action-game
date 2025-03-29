#include "radiant.h"

#include "optix/vec_math.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cassert>
#include <string>
#include <sstream>
#include <vector>

#include "lm_bake.h"

#include "optixir_binary.h"

// Simple error checking macros for CUDA and OptiX calls.
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = call;                                          \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << "\n"; \
            assert(0);                                                       \
        }                                                                  \
    } while (0)

#define OPTIX_CHECK(call)                                                  \
    do {                                                                   \
        OptixResult res = call;                                            \
        if (res != OPTIX_SUCCESS) {                                        \
            std::cerr << "Optix Error: " << optixGetErrorString(res) << "\n"; \
            assert(0);                                                       \
        }                                                                  \
    } while (0)

inline void optixCheckLog(OptixResult  res,
    const char *log,
    size_t       sizeof_log,
    size_t       sizeof_log_returned,
    const char *call,
    const char *file,
    unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        std::cerr << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
            << log << (sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "") << '\n';
        assert(0);
    }
}

// This version of the log-check macro doesn't require the user do setup
// a log buffer and size variable in the surrounding context; rather the
// macro defines a log buffer and log size variable (LOG and LOG_SIZE)
// respectively that should be passed to the message being checked.
// E.g.:
//  OPTIX_CHECK_LOG2( optixProgramGroupCreate( ..., LOG, &LOG_SIZE, ... );
//
#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        char   LOG[2048];                                                      \
        size_t LOG_SIZE = sizeof( LOG );                                       \
        optixCheckLog( call, LOG, sizeof( LOG ), LOG_SIZE, #call,              \
                                __FILE__, __LINE__ );                          \
    } while( false )

inline void cudaSyncCheck(const char *file, unsigned int line)
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA error on synchronize with error '"
            << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
        assert(0);
    }
}

#define CUDA_SYNC_CHECK() cudaSyncCheck( __FILE__, __LINE__ )

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;


namespace sutil {

    // implementing a perspective camera
    class Camera {
    public:
        Camera()
            : m_eye(make_float3(1.0f)), m_lookat(make_float3(0.0f)), m_up(make_float3(0.0f, 1.0f, 0.0f)), m_fovY(35.0f), m_aspectRatio(1.0f)
        {
        }

        Camera(const float3 &eye, const float3 &lookat, const float3 &up, float fovY, float aspectRatio)
            : m_eye(eye), m_lookat(lookat), m_up(up), m_fovY(fovY), m_aspectRatio(aspectRatio)
        {
        }

        float3 direction() const { return normalize(m_lookat - m_eye); }
        void setDirection(const float3 &dir) { m_lookat = m_eye + length(m_lookat - m_eye) * dir; }

        const float3 &eye() const { return m_eye; }
        void setEye(const float3 &val) { m_eye = val; }
        const float3 &lookat() const { return m_lookat; }
        void setLookat(const float3 &val) { m_lookat = val; }
        const float3 &up() const { return m_up; }
        void setUp(const float3 &val) { m_up = val; }
        const float &fovY() const { return m_fovY; }
        void setFovY(const float &val) { m_fovY = val; }
        const float &aspectRatio() const { return m_aspectRatio; }
        void setAspectRatio(const float &val) { m_aspectRatio = val; }

        // UVW forms an orthogonal, but not orthonormal basis!
        void UVWFrame(float3 &U, float3 &V, float3 &W) const
        {
            W = m_lookat - m_eye; // Do not normalize W -- it implies focal length
            float wlen = length(W);
            U = normalize(cross(W, m_up));
            V = normalize(cross(U, W));

            float vlen = wlen * tanf(0.5f * m_fovY * M_PIf / 180.0f);
            V *= vlen;
            float ulen = vlen * m_aspectRatio;
            U *= ulen;
        }

    private:
        float3 m_eye;
        float3 m_lookat;
        float3 m_up;
        float m_fovY;
        float m_aspectRatio;
    };

}

namespace sutil
{
    template <typename PIXEL_FORMAT>
    class CUDAOutputBuffer
    {
    public:
        CUDAOutputBuffer(int32_t width, int32_t height);
        ~CUDAOutputBuffer();

        void resize(int32_t width, int32_t height);

        // Allocate or update device pointer as necessary for CUDA access
        PIXEL_FORMAT *map();
        void unmap();

        int32_t        width() const { return m_width; }
        int32_t        height() const { return m_height; }

        PIXEL_FORMAT   *getHostPointer();

    private:
        void makeCurrent() { CUDA_CHECK(cudaSetDevice(m_device_idx)); }

        int32_t                    m_width = 0u;
        int32_t                    m_height = 0u;

        PIXEL_FORMAT *m_device_pixels = nullptr;
        std::vector<PIXEL_FORMAT>  m_host_pixels;

        CUstream                   m_stream = 0u;
        int32_t                    m_device_idx = 0;
    };


    template <typename PIXEL_FORMAT>
    CUDAOutputBuffer<PIXEL_FORMAT>::CUDAOutputBuffer(int32_t width, int32_t height)
    {
        // Output dimensions must be at least 1 in both x and y to avoid an error
        // with cudaMalloc.

        if (width <= 0)
            width = 1;
        if (height <= 0)
            height = 1;

        resize(width, height);
    }

    template <typename PIXEL_FORMAT>
    CUDAOutputBuffer<PIXEL_FORMAT>::~CUDAOutputBuffer()
    {
        makeCurrent();
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_device_pixels)));
    }


    template <typename PIXEL_FORMAT>
    void CUDAOutputBuffer<PIXEL_FORMAT>::resize(int32_t width, int32_t height)
    {
        // Output dimensions must be at least 1 in both x and y to avoid an error
        // with cudaMalloc.
        if (width <= 0)
            width = 1;
        if (height <= 0)
            height = 1;

        if (m_width == width && m_height == height)
            return;

        m_width = width;
        m_height = height;

        makeCurrent();

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_device_pixels)));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&m_device_pixels),
            m_width * m_height * sizeof(PIXEL_FORMAT)
        ));

        if (!m_host_pixels.empty())
            m_host_pixels.resize(m_width * m_height);
    }

    template <typename PIXEL_FORMAT>
    PIXEL_FORMAT *CUDAOutputBuffer<PIXEL_FORMAT>::map()
    {
        return m_device_pixels;
    }

    template <typename PIXEL_FORMAT>
    void CUDAOutputBuffer<PIXEL_FORMAT>::unmap()
    {
        makeCurrent();
        CUDA_CHECK(cudaStreamSynchronize(m_stream));
    }

    template <typename PIXEL_FORMAT>
    PIXEL_FORMAT *CUDAOutputBuffer<PIXEL_FORMAT>::getHostPointer()
    {
        m_host_pixels.resize(m_width * m_height);

        makeCurrent();
        CUDA_CHECK(cudaMemcpy(
            static_cast<void *>(m_host_pixels.data()),
            map(),
            m_width * m_height * sizeof(PIXEL_FORMAT),
            cudaMemcpyDeviceToHost
        ));
        unmap();

        return m_host_pixels.data();
    }

} // end namespace sutil

extern "C" RADIANT_API void __cdecl RadiantBake(radiant_bake_info_t BakeInfo)
{
    //
    // Initialize CUDA and create OptiX context
    //
    OptixDeviceContext context = nullptr;
    {
        // Initialize CUDA
        CUDA_CHECK(cudaFree(0));

        // Initialize the OptiX API, loading all API entry points
        OPTIX_CHECK(optixInit());

        // Specify context options
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = 0;
        options.logCallbackLevel = 4;

        // Associate a CUDA context (and therefore a specific GPU) with this
        // device context
        CUcontext cuCtx = 0;  // zero means take the current context
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }


    //
    // accel handling
    //
    OptixTraversableHandle gas_handle;
    CUdeviceptr            d_gas_output_buffer;
    {
        // Use default options for simplicity.  In a real use case we would want to
        // enable compaction, etc
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        // Triangle build input: simple list of three vertices
        // const std::array<float3, 3> vertices =
        // { {
        //       { -0.5f, -0.5f, 0.0f },
        //       {  0.5f, -0.5f, 0.0f },
        //       {  0.0f,  0.5f, 0.0f }
        // } };

        const size_t vertices_size = sizeof(radiant_vec3_t) * BakeInfo.WorldGeometryVerticesCount;
        CUdeviceptr d_vertices = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(d_vertices),
            BakeInfo.WorldGeometryVertices,
            vertices_size,
            cudaMemcpyHostToDevice
        ));

        // Our build input is a simple list of non-indexed triangle vertices
        const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices = static_cast<uint32_t>(BakeInfo.WorldGeometryVerticesCount);
        triangle_input.triangleArray.vertexBuffers = &d_vertices;
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &accel_options,
            &triangle_input,
            1, // Number of build inputs
            &gas_buffer_sizes
        ));
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_temp_buffer_gas),
            gas_buffer_sizes.tempSizeInBytes
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_gas_output_buffer),
            gas_buffer_sizes.outputSizeInBytes
        ));

        OPTIX_CHECK(optixAccelBuild(
            context,
            0,                  // CUDA stream
            &accel_options,
            &triangle_input,
            1,                  // num build inputs
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            d_gas_output_buffer,
            gas_buffer_sizes.outputSizeInBytes,
            &gas_handle,
            nullptr,            // emitted property list
            0                   // num emitted properties
        ));

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));
    }

    //
    // Create module
    //
    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};
#ifndef NDEBUG
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif // NDEBUG
        // NOTE(Kevin): holy fuck compiling the optixir in Release mode and setting
        // optLevel to all optimizations and debug level to minimal (the defaults) 
        // makes it so fast. 634375 texels (at texel size 2) with 4096 sample rays per
        // texel is like 1.5 seconds... 

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 14; // TRICKY!
        pipeline_compile_options.numAttributeValues = 3;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "Params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        OPTIX_CHECK_LOG(optixModuleCreate(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            (char*)OptixIRBinary,
            OptixIRBinarySize,
            LOG, &LOG_SIZE,
            &module
        ));
    }

    //
    // Create program groups
    //
    OptixProgramGroup raygen_prog_group           = 0;
    OptixProgramGroup directionallight_miss_group = 0;
    OptixProgramGroup pointlight_miss_group       = 0;
    OptixProgramGroup HemisphereSample_miss_group = 0;
    OptixProgramGroup directionallight_hit_group  = 0;
    OptixProgramGroup pointlight_hit_group        = 0;
    OptixProgramGroup HemisphereSample_hit_group  = 0;
    {
        OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc = {}; //
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &raygen_prog_group
        ));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__DirectionalLight";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &directionallight_miss_group
        ));

        memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__PointLight";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &pointlight_miss_group
        ));

        memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__HemisphereSample";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &HemisphereSample_miss_group
        ));

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__DirectionalLight";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &hitgroup_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &directionallight_hit_group
        ));

        memset(&hitgroup_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__PointLight";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &hitgroup_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &pointlight_hit_group
        ));

        memset(&hitgroup_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__HemisphereSample";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &hitgroup_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &HemisphereSample_hit_group
        ));
    }

    //
    // Link pipeline
    //
    OptixPipeline pipeline = nullptr;
    {
        const uint32_t    max_trace_depth = 3;
        OptixProgramGroup program_groups[] = { 
            raygen_prog_group, 
            directionallight_miss_group,
            pointlight_miss_group,
            HemisphereSample_miss_group,
            directionallight_hit_group,
            pointlight_hit_group,
            HemisphereSample_hit_group
        };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        OPTIX_CHECK_LOG(optixPipelineCreate(
            context,
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups) / sizeof(program_groups[0]),
            LOG, &LOG_SIZE,
            &pipeline
        ));

        OptixStackSizes stack_sizes = {};
        for (auto &prog_group : program_groups)
        {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
            0,  // maxCCDepth
            0,  // maxDCDEpth
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state, continuation_stack_size,
            1  // maxTraversableDepth
        ));
    }

    //
    // Set up shader binding table
    //
    OptixShaderBindingTable sbt = {};
    {
        CUdeviceptr  raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(raygen_record),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice
        ));


        CUdeviceptr miss_record;
        size_t      miss_record_size = sizeof(MissSbtRecord) * RAY_TYPE_COUNT;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));

        MissSbtRecord ms_sbt[RAY_TYPE_COUNT];
        OPTIX_CHECK(optixSbtRecordPackHeader(directionallight_miss_group, &ms_sbt[RAY_TYPE_DIRECTIONAL_LIGHT]));
        OPTIX_CHECK(optixSbtRecordPackHeader(pointlight_miss_group, &ms_sbt[RAY_TYPE_POINT_LIGHT]));
        OPTIX_CHECK(optixSbtRecordPackHeader(HemisphereSample_miss_group, &ms_sbt[RAY_TYPE_HEMISPHERE_SAMPLE]));

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(miss_record),
            &ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice
        ));


        CUdeviceptr hitgroup_record;
        size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord) * RAY_TYPE_COUNT;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));

        HitGroupSbtRecord hg_sbt[RAY_TYPE_COUNT];
        OPTIX_CHECK(optixSbtRecordPackHeader(directionallight_hit_group, &hg_sbt[RAY_TYPE_DIRECTIONAL_LIGHT]));
        OPTIX_CHECK(optixSbtRecordPackHeader(pointlight_hit_group, &hg_sbt[RAY_TYPE_POINT_LIGHT]));
        OPTIX_CHECK(optixSbtRecordPackHeader(HemisphereSample_hit_group, &hg_sbt[RAY_TYPE_HEMISPHERE_SAMPLE]));

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(hitgroup_record),
            &hg_sbt,
            hitgroup_record_size,
            cudaMemcpyHostToDevice
        ));


        sbt.raygenRecord = raygen_record;
        sbt.missRecordBase = miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = RAY_TYPE_COUNT;
        sbt.hitgroupRecordBase = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = RAY_TYPE_COUNT;
    }


    // setting up input output buffers

    CUdeviceptr *d_world_positions;
    CUDA_CHECK(cudaMalloc(&d_world_positions, BakeInfo.OutputLightmapSize * sizeof(float3))); // Allocate memory on the device
    CUDA_CHECK(cudaMemcpy(d_world_positions, BakeInfo.LightMapTexelPositions, 
        BakeInfo.OutputLightmapSize * sizeof(float3), cudaMemcpyHostToDevice));
    CUdeviceptr *d_world_normals;
    CUDA_CHECK(cudaMalloc(&d_world_normals, BakeInfo.OutputLightmapSize * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_world_normals, BakeInfo.LightMapTexelNormals, 
        BakeInfo.OutputLightmapSize * sizeof(float3), cudaMemcpyHostToDevice));

    int PointLightsCount = (int)BakeInfo.PointLightsCount;
    cu_pointlight_t *PointLightInfos = (cu_pointlight_t*)malloc(sizeof(cu_pointlight_t) * PointLightsCount);
    for (int i = 0; i < PointLightsCount; ++i)
    {
        radiant_pointlight_t PLight = BakeInfo.PointLights[i];

        PointLightInfos[i].Position = *((float3*)&PLight.Position);
        PointLightInfos[i].AttenuationLinear = PLight.AttenuationLinear;
        PointLightInfos[i].AttenuationQuadratic = PLight.AttenuationQuadratic;
    }

    CUdeviceptr *d_point_light_srcs;
    CUDA_CHECK(cudaMalloc(&d_point_light_srcs, PointLightsCount * sizeof(cu_pointlight_t)));
    if (PointLightsCount > 0)
    {
        CUDA_CHECK(cudaMemcpy(d_point_light_srcs, PointLightInfos, PointLightsCount * sizeof(cu_pointlight_t), cudaMemcpyHostToDevice));
    }
    free(PointLightInfos);

    bool NoSun = BakeInfo.DirectionToSun.x == 0.f && BakeInfo.DirectionToSun.y == 0.f && BakeInfo.DirectionToSun.z == 0.f;
    float3 DirectionToSun = *((float3*)&BakeInfo.DirectionToSun);
    DirectionToSun = normalize(DirectionToSun);


    //
    // launch
    //

    // 1. Lightmap baking
    sutil::CUDAOutputBuffer<float> LightmapBuffer(BakeInfo.OutputLightmapSize, 1);
    {
        CUstream stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        bake_lm_params_t BakeParams;
        BakeParams.Procedure = BAKE_LIGHTMAP;
        BakeParams.OutputLightmap = LightmapBuffer.map();
        BakeParams.TexelWorldPositions = (float3*)d_world_positions;
        BakeParams.TexelWorldNormals = (float3*)d_world_normals;

        BakeParams.DoSunLight = int(!NoSun);
        BakeParams.DirectionToSun = *((float3*)&DirectionToSun);
        BakeParams.SkyboxColor = *((float3*)&BakeInfo.SkyboxColor);
        BakeParams.SkyboxBrightness = BakeInfo.SkyboxBrightness;
        BakeParams.CountOfPointLights = PointLightsCount;
        BakeParams.PointLights = (cu_pointlight_t*)d_point_light_srcs;
        BakeParams.GASHandle = gas_handle;
        BakeParams.NumberOfSampleRaysPerTexel = BakeInfo.NumberOfSampleRaysPerTexel;
        BakeParams.NumberOfBounces = BakeInfo.NumberOfLightBounces;
        BakeParams.BakeDirectLighting = int(BakeInfo.BakeDirectLighting);

        CUdeviceptr d_param;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(bake_lm_params_t)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(d_param),
            &BakeParams, sizeof(BakeParams),
            cudaMemcpyHostToDevice
        ));

        OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(bake_lm_params_t), &sbt, BakeInfo.OutputLightmapSize, 1, /*depth=*/1));
        CUDA_SYNC_CHECK();

        LightmapBuffer.unmap();
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param)));
    }
    memcpy(BakeInfo.OutputLightmap, LightmapBuffer.getHostPointer(), BakeInfo.OutputLightmapSize*sizeof(float));

    // 2. Caching direct lights that pass through light caches
    if (BakeInfo.CacheDirectLightIndices)
    {
        size_t NumLightCaches = BakeInfo.OutputDirectLightIndicesSize / BakeInfo.OutputDirectLightIndicesPerSample;
        CUdeviceptr *d_lightcache_world_positions;
        CUDA_CHECK(cudaMalloc(&d_lightcache_world_positions, NumLightCaches * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(d_lightcache_world_positions, BakeInfo.DirectLightCachePositions,
            NumLightCaches * sizeof(float3), cudaMemcpyHostToDevice));

        sutil::CUDAOutputBuffer<short> LightIndicesBuffer(BakeInfo.OutputDirectLightIndicesSize, 1);
        {
            CUstream stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            bake_lm_params_t BakeParams;
            BakeParams.Procedure = BAKE_DIRECTLIGHTINFO;
            BakeParams.OutputDirectLightIndices = LightIndicesBuffer.map();
            BakeParams.OutputDirectLightIndicesPerSample = BakeInfo.OutputDirectLightIndicesPerSample;
            BakeParams.DirectLightCachePositions = (float3*)d_lightcache_world_positions;

            BakeParams.DoSunLight = int(!NoSun);
            BakeParams.DirectionToSun = *((float3*)&DirectionToSun);
            BakeParams.SkyboxColor = *((float3*)&BakeInfo.SkyboxColor);
            BakeParams.SkyboxBrightness = BakeInfo.SkyboxBrightness;
            BakeParams.CountOfPointLights = PointLightsCount;
            BakeParams.PointLights = (cu_pointlight_t*)d_point_light_srcs;
            BakeParams.GASHandle = gas_handle;
            BakeParams.NumberOfSampleRaysPerTexel = 0;
            BakeParams.NumberOfBounces = 0;
            BakeParams.BakeDirectLighting = 0;

            CUdeviceptr d_param;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(bake_lm_params_t)));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(d_param),
                &BakeParams, sizeof(BakeParams),
                cudaMemcpyHostToDevice
            ));

            OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(bake_lm_params_t), &sbt, BakeInfo.OutputLightmapSize, 1, /*depth=*/1));
            CUDA_SYNC_CHECK();

            LightIndicesBuffer.unmap();
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param)));
        }
        memcpy(BakeInfo.OutputDirectLightIndices, LightIndicesBuffer.getHostPointer(), BakeInfo.OutputDirectLightIndicesSize*sizeof(short));
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_world_positions)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_world_normals)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_point_light_srcs)));


    //
    // Cleanup
    //
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.missRecordBase)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_gas_output_buffer)));

        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        OPTIX_CHECK(optixProgramGroupDestroy(directionallight_hit_group));
        OPTIX_CHECK(optixProgramGroupDestroy(directionallight_miss_group));
        OPTIX_CHECK(optixProgramGroupDestroy(pointlight_hit_group));
        OPTIX_CHECK(optixProgramGroupDestroy(pointlight_miss_group));
        OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
        OPTIX_CHECK(optixModuleDestroy(module));

        OPTIX_CHECK(optixDeviceContextDestroy(context));
    }
}

