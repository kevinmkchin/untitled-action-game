#include "physics.h"


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
static bool AssertFailedImpl(const char *inExpression, const char *inMessage, const char *inFile, uint inLine)
{
    // Print to the TTY
    std::cout << inFile << ":" << inLine << ": (" << inExpression << ") " << (inMessage != nullptr? inMessage : "") << std::endl;

    // Breakpoint
    return true;
};

#endif // JPH_ENABLE_ASSERTS


bool obj_layer_pair_filter_t::ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const
{
    switch (inObject1)
    {
        case Layers::NON_MOVING:
            return inObject2 == Layers::MOVING; // Non moving only collides with moving
        case Layers::MOVING:
            return true; // Moving collides with everything
        default:
            JPH_ASSERT(false);
            return false;
    }
}

bool obj_vs_bp_layer_impl_t::ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const
{
    switch (inLayer1)
    {
        case Layers::NON_MOVING:
            return inLayer2 == BroadPhaseLayers::MOVING;
        case Layers::MOVING:
            return true;
        default:
            JPH_ASSERT(false);
            return false;
    }
}

void physics_t::Initialize()
{
    // Register allocation hook. In this example we'll just let Jolt use
    // malloc / free but you can override these if you want (see Memory.h).
    // This needs to be done before any other Jolt function is called.
    JPH::RegisterDefaultAllocator();
    // Install trace and assert callbacks
    JPH::Trace = TraceImpl;
    JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = AssertFailedImpl;)

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
    const uint cMaxBodies = 65536;

    // This determines how many mutexes to allocate to protect rigid bodies 
    // from concurrent access. Set it to 0 for the default settings.
    const uint cNumBodyMutexes = 0;

    // This is the max amount of body pairs that can be queued at any time 
    // (the broad phase will detect overlapping body pairs based on their 
    // bounding boxes and will insert them into a queue for the narrowphase).
    // If you make this buffer too small the queue will fill up and the broad 
    // phase jobs will start to do narrow phase work. This is slightly less efficient.
    // Note: This value is low because this is a simple test. For a real project 
    // use something in the order of 65536.
    const uint cMaxBodyPairs = 65536;

    // This is the maximum size of the contact constraint buffer. If more contacts
    // (collisions between bodies) are detected than this number then these contacts
    // will be ignored and bodies will start interpenetrating / fall through the world.
    // Note: For a real project use something in the order of 10240.
    const uint cMaxContactConstraints = 10240;

    // Now we can create the actual physics system.
    PhysicsSystem = new JPH::PhysicsSystem();
    PhysicsSystem->Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints, 
        BroadPhaseLayerInterface, ObjectVsBroadphaseFilter, ObjectVsObjectFilter);

    PhysicsSystem->SetGravity(PhysicsSystem->GetGravity() * 32.f);

    // The main way to interact with the bodies in the physics system is through 
    // the body interface. There is a locking and a non-locking variant of this. 
    // We're going to use the locking version (even though we're not planning to 
    // access bodies from multiple threads).
    BodyInterface = &PhysicsSystem->GetBodyInterface();

}

void physics_t::Destroy()
{
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
    static JPH::TempAllocatorImpl PhysTempAllocator(10 * 1024 * 1024);
    static JPH::JobSystemThreadPool PhysJobSystem(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, 
        std::thread::hardware_concurrency() - 1);

    // NOTE(Kevin 2025-01-30): Should keep this fixed. Global dt can be huge when program spends a lot of time
    //                         doing something like building a map. If it's fixed, then physics/world just simulates
    //                         slower if framerate drops below 60.
    const float cDeltaTime = FixedDeltaTime;//1.0f / 60.0f;
    const int cCollisionSteps = 1;

    PhysicsSystem->Update(cDeltaTime, cCollisionSteps, &PhysTempAllocator, &PhysJobSystem);
}

JPH::RVec3 ToJoltVec3(vec3 GMathVec3)
{
    return JPH::RVec3(GMathVec3.x, GMathVec3.y, GMathVec3.z);
}

vec3 FromJoltVec3(JPH::RVec3 JoltVec3)
{
    return vec3(JoltVec3.GetX(), JoltVec3.GetY(), JoltVec3.GetZ());
}

JPH::Quat ToJoltQuat(quat GMathQuat)
{
    return JPH::Quat(GMathQuat.x, GMathQuat.y, GMathQuat.z, GMathQuat.w);
}

quat FromJoltQuat(JPH::Quat JoltQuat)
{
    return quat(JoltQuat.GetW(), JoltQuat.GetX(), JoltQuat.GetY(), JoltQuat.GetZ());
}

