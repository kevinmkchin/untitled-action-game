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

void physics_t::Update()
{
    static JPH::TempAllocatorImpl PhysTempAllocator(10 * 1024 * 1024);
    static JPH::JobSystemThreadPool PhysJobSystem(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, 
        std::thread::hardware_concurrency() - 1);

    const float cDeltaTime = 1.0f / 60.0f;
    const int cCollisionSteps = 1;

    // Step the world
    PhysicsSystem->Update(cDeltaTime, cCollisionSteps, &PhysTempAllocator, &PhysJobSystem);
}

// void HelloWorld()
// {

//     // We need a temp allocator for temporary allocations during the physics update. We're
//     // pre-allocating 10 MB to avoid having to do allocations during the physics update.
//     // B.t.w. 10 MB is way too much for this example but it is a typical value you can use.
//     // If you don't want to pre-allocate you can also use TempAllocatorMalloc to fall back to
//     // malloc / free.
//     TempAllocatorImpl temp_allocator(10 * 1024 * 1024);

//     // We need a job system that will execute physics jobs on multiple threads. Typically
//     // you would implement the JobSystem interface yourself and let Jolt Physics run on top
//     // of your own job scheduler. JobSystemThreadPool is an example implementation.
//     JobSystemThreadPool job_system(cMaxPhysicsJobs, cMaxPhysicsBarriers, thread::hardware_concurrency() - 1);


//     // Next we can create a rigid body to serve as the floor, we make a large box
//     // Create the settings for the collision volume (the shape).
//     // Note that for simple shapes (like boxes) you can also directly construct a BoxShape.
//     BoxShapeSettings floor_shape_settings(Vec3(100.0f, 1.0f, 100.0f));
//     floor_shape_settings.SetEmbedded(); // A ref counted object on the stack (base class RefTarget) should be marked as such to prevent it from being freed when its reference count goes to 0.

//     // Create the shape
//     ShapeSettings::ShapeResult floor_shape_result = floor_shape_settings.Create();
//     ShapeRefC floor_shape = floor_shape_result.Get(); // We don't expect an error here, but you can check floor_shape_result for HasError() / GetError()

//     // Create the settings for the body itself. Note that here you can also set other properties like the restitution / friction.
//     BodyCreationSettings floor_settings(floor_shape, RVec3(0.0, -1.0, 0.0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);

//     // Create the actual rigid body
//     Body *floor = body_interface.CreateBody(floor_settings); // Note that if we run out of bodies this can return nullptr

//     // Add it to the world
//     body_interface.AddBody(floor->GetID(), EActivation::DontActivate);

//     // Now create a dynamic body to bounce on the floor
//     // Note that this uses the shorthand version of creating and adding a body to the world
//     BodyCreationSettings sphere_settings(new SphereShape(0.5f), RVec3(0.0, 2.0, 0.0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING);
//     BodyID sphere_id = body_interface.CreateAndAddBody(sphere_settings, EActivation::Activate);

//     // Now you can interact with the dynamic body, in this case we're going to give it a velocity.
//     // (note that if we had used CreateBody then we could have set the velocity straight on the body before adding it to the physics system)
//     body_interface.SetLinearVelocity(sphere_id, Vec3(0.0f, -5.0f, 0.0f));

//     // We simulate the physics world in discrete time steps. 60 Hz is a good rate to update the physics system.
//     const float cDeltaTime = 1.0f / 60.0f;

//     // Optional step: Before starting the physics simulation you can optimize the broad phase. This improves collision detection performance (it's pointless here because we only have 2 bodies).
//     // You should definitely not call this every frame or when e.g. streaming in a new level section as it is an expensive operation.
//     // Instead insert all new objects in batches instead of 1 at a time to keep the broad phase efficient.
//     physics_system.OptimizeBroadPhase();

//     // Now we're ready to simulate the body, keep simulating until it goes to sleep
//     uint step = 0;
//     while (body_interface.IsActive(sphere_id))
//     {
//         // Next step
//         ++step;

//         // Output current position and velocity of the sphere
//         RVec3 position = body_interface.GetCenterOfMassPosition(sphere_id);
//         Vec3 velocity = body_interface.GetLinearVelocity(sphere_id);
//         std::cout << "Step " << step << ": Position = (" << position.GetX() << ", " << position.GetY() << ", " << position.GetZ() << "), Velocity = (" << velocity.GetX() << ", " << velocity.GetY() << ", " << velocity.GetZ() << ")" << std::endl;

//         // If you take larger steps than 1 / 60th of a second you need to do multiple collision steps in order to keep the simulation stable. Do 1 collision step per 1 / 60th of a second (round up).
//         const int cCollisionSteps = 1;

//         // Step the world
//         physics_system.Update(cDeltaTime, cCollisionSteps, &temp_allocator, &job_system);
//     }

//     // Remove the sphere from the physics system. Note that the sphere itself keeps all of its state and can be re-added at any time.
//     body_interface.RemoveBody(sphere_id);

//     // Destroy the sphere. After this the sphere ID is no longer valid.
//     body_interface.DestroyBody(sphere_id);

//     // Remove and destroy the floor
//     body_interface.RemoveBody(floor->GetID());
//     body_interface.DestroyBody(floor->GetID());
// }
