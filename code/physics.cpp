#include "physics.h"
#include "mem.h"


// external
physics_t Physics;


// Custom memory management
static void *CustomJPHAllocate(size_t inSize)
{
    JPH_ASSERT(inSize > 0);
    return JoltPhysicsMemory.alloc(inSize);
}

static void *CustomJPHReallocate(void *inBlock, size_t inOldSize, size_t inNewSize)
{
    JPH_ASSERT(inNewSize > 0);
    return JoltPhysicsMemory.realloc(inBlock, inNewSize);
}

static void CustomJPHFree(void *inBlock)
{
    JoltPhysicsMemory.free(inBlock);
}

static void *CustomJPHAlignedAllocate(size_t inSize, size_t inAlignment)
{
    JPH_ASSERT(inSize > 0 && inAlignment > 0);
    return JoltPhysicsMemory.alloc(inSize, inAlignment);
}

static void CustomJPHAlignedFree(void *inBlock)
{
    JoltPhysicsMemory.free(inBlock);
}


// Disable common warnings triggered by Jolt, you can use JPH_SUPPRESS_WARNING_PUSH / JPH_SUPPRESS_WARNING_POP to store and restore the warning state
JPH_SUPPRESS_WARNINGS

// Callback for traces, connect this to your own trace function if you have one
static void TraceImpl(const char *inFMT, ...)
{
    // Format the message
    va_list list;
    va_start(list, inFMT);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), inFMT, list);
    va_end(list);

    // Print to the TTY
    std::cout << buffer << std::endl;
}

#ifdef JPH_ENABLE_ASSERTS
// Callback for asserts, connect this to your own assert handler if you have one
static bool AssertFailedImpl(const char *inExpression, const char *inMessage, const char *inFile, unsigned int inLine)
{
    // Print to the TTY
    std::cout << inFile << ":" << inLine << ": (" << inExpression << ") " << (inMessage != nullptr? inMessage : "") << std::endl;

    // Breakpoint
    return true;
};
#endif // JPH_ENABLE_ASSERTS

#include "physics_bodies.cpp"

void physics_t::Initialize()
{
    // Register allocation hook. This needs to be done before any other Jolt function is called.
    JPH::Allocate = CustomJPHAllocate;
    JPH::Reallocate = CustomJPHReallocate;
    JPH::Free = CustomJPHFree;
    JPH::AlignedAllocate = CustomJPHAlignedAllocate;
    JPH::AlignedFree = CustomJPHAlignedFree;

    // Install trace and assert callbacks
    JPH::Trace = TraceImpl;
    JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = AssertFailedImpl;)

    TempAllocator = new JPH::TempAllocatorImpl(10 * 1024 * 1024);

    // Create a factory, this class is responsible for creating instances 
    // of classes based on their name or hash and is mainly used for 
    // deserialization of saved data. It is not directly used in this 
    // example but still required.
    JPH::Factory::sInstance = new JPH::Factory();

    // Register all physics types with the factory and install their collision
    // handlers with the CollisionDispatch class. If you have your own custom 
    // shape types you probably need to register their handlers with the 
    // CollisionDispatch before calling this function. If you implement your
    // own default material (PhysicsMaterial::sDefault) make sure to initialize 
    // it before this function or else this function will create one for you.
    JPH::RegisterTypes();

    // This is the max amount of rigid bodies that you can add to the physics 
    // system. If you try to add more you'll get an error. Note: For a real 
    // project use something in the order of 65536.
    const unsigned int cMaxBodies = 65536;

    // This determines how many mutexes to allocate to protect rigid bodies 
    // from concurrent access. Set it to 0 for the default settings.
    const unsigned int cNumBodyMutexes = 0;

    // This is the max amount of body pairs that can be queued at any time 
    // (the broad phase will detect overlapping body pairs based on their 
    // bounding boxes and will insert them into a queue for the narrowphase).
    // If you make this buffer too small the queue will fill up and the broad 
    // phase jobs will start to do narrow phase work. This is slightly less efficient.
    // Note: This value is low because this is a simple test. For a real project 
    // use something in the order of 65536.
    const unsigned int cMaxBodyPairs = 65536;

    // This is the maximum size of the contact constraint buffer. If more contacts
    // (collisions between bodies) are detected than this number then these contacts
    // will be ignored and bodies will start interpenetrating / fall through the world.
    // Note: For a real project use something in the order of 10240.
    const unsigned int cMaxContactConstraints = 10240;

    ObjectLayerFilter = CreateAndSetupObjectLayers();
    BroadphaseMapping = CreateAndSetupBroadPhaseLayers();
    ObjectVsBroadphaseFilter = CreateAndSetupObjectVsBroadPhaseFilter(BroadphaseMapping, ObjectLayerFilter);

    // Now we can create the actual physics system.
    PhysicsSystem = new JPH::PhysicsSystem();
    PhysicsSystem->Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints, 
        *BroadphaseMapping, *ObjectVsBroadphaseFilter, *ObjectLayerFilter);

    // A contact listener gets notified when bodies (are about to) collide, and when they separate again.
    // Note that this is called from a job so whatever you do here needs to be thread safe.
    // Registering one is entirely optional.
    PhysicsSystem->SetContactListener(&ContactListener);

    PhysicsSystem->SetGravity(JPH::Vec3(0, -9.81f, 0));

    // The main way to interact with the bodies in the physics system is through 
    // the body interface. There is a locking and a non-locking variant of this. 
    // We're going to use the locking version (even though we're not planning to 
    // access bodies from multiple threads).
    BodyInterface = &PhysicsSystem->GetBodyInterface();

}

void physics_t::Destroy()
{
    delete TempAllocator;

    delete PhysicsSystem;
    PhysicsSystem = nullptr;
    BodyInterface = nullptr;

    // Unregisters all types with the factory and cleans up the default material
    JPH::UnregisterTypes();

    // Destroy the factory
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;
}

void physics_t::Tick()
{
    static JPH::JobSystemThreadPool PhysJobSystem(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, 
        std::thread::hardware_concurrency() - 1);

    // NOTE(Kevin 2025-01-30): Should keep this fixed. Global dt can be huge when program spends a lot of time
    //                         doing something like building a map. If it's fixed, then physics/world just simulates
    //                         slower if framerate drops below 60.
    const float cDeltaTime = FixedDeltaTime;//1.0f / 60.0f;
    const int cCollisionSteps = 1;

    PhysicsSystem->Update(cDeltaTime, cCollisionSteps, TempAllocator, &PhysJobSystem);
}


