#pragma once

// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

// The Jolt headers don't include Jolt.h. Always include Jolt.h before including any other Jolt header.
// You can use Jolt.h in your precompiled header to speed up compilation.
#if INTERNAL_BUILD
#define JPH_DEBUG_RENDERER
#endif // INTERNAL_BUILD
#include <Jolt/Jolt.h>
// Jolt includes
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionDispatch.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Character/Character.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
// STL includes
#include <iostream>
#include <thread>
#include <cstdarg>

using uint = unsigned int;
using body_id = JPH::BodyID;

// Layer that objects can be in, determines which other objects it can collide with
// Typically you at least want to have 1 layer for moving bodies and 1 layer for static bodies,
// but you can have more layers if you want. E.g. you could have a layer for high detail 
// collision (which is not used by the physics simulation but only if you do collision testing).
namespace Layers
{
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING = 1;
    static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
};

// Class that determines if two object layers can collide
// Note: As this is an interface, JPH::PhysicsSystem will take a reference to this so this instance 
// needs to stay alive! Also have a look at ObjectLayerPairFilterTable or ObjectLayerPairFilterMask 
// for a simpler interface.
class obj_layer_pair_filter_t : public JPH::ObjectLayerPairFilter
{
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override;
};

// Each broadphase layer results in a separate bounding volume tree in the broad phase. You at
// least want to have a layer for non-moving and moving objects to avoid having to update a tree
// full of static objects every frame. You can have a 1-on-1 mapping between object layers and 
// broadphase layers (like in this case) but if you have many object layers you'll be creating 
// many broad phase trees, which is not efficient. If you want to fine tune your broadphase layers
// define JPH_TRACK_BROADPHASE_STATS and look at the stats reported on the TTY.
namespace BroadPhaseLayers
{
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr unsigned int NUM_LAYERS(2);
};

// BroadPhaseLayerInterface implementation
// This defines a mapping table between object and broadphase layers.
// Note: As this is an interface, JPH::PhysicsSystem will take a reference to this so this 
// instance needs to stay alive! Also have a look at BroadPhaseLayerInterfaceTable or 
// BroadPhaseLayerInterfaceMask for a simpler interface.
class bp_layer_interface_t final : public JPH::BroadPhaseLayerInterface
{
public:
    bp_layer_interface_t()
    {
        // Create a mapping table from object to broad phase layer
        mObjectToBroadPhase[Layers::NON_MOVING] = BroadPhaseLayers::NON_MOVING;
        mObjectToBroadPhase[Layers::MOVING] = BroadPhaseLayers::MOVING;
    }

    virtual unsigned int GetNumBroadPhaseLayers() const override
    {
        return BroadPhaseLayers::NUM_LAYERS;
    }

    virtual JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override
    {
        JPH_ASSERT(inLayer < Layers::NUM_LAYERS);
        return mObjectToBroadPhase[inLayer];
    }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    virtual const char *GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const override
    {
        switch ((BroadPhaseLayer::Type)inLayer)
        {
            case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING: 
                return "NON_MOVING";
            case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::MOVING: 
                return "MOVING";
            default:
                JPH_ASSERT(false); return "INVALID";
        }
    }
#endif // JPH_EXTERNAL_PROFILE || JPH_PROFILE_ENABLED

private:
    JPH::BroadPhaseLayer mObjectToBroadPhase[Layers::NUM_LAYERS];
};

// Class that determines if an object layer can collide with a broadphase layer
// Note: As this is an interface, PhysicsSystem will take a reference to this so this
// instance needs to stay alive! Also have a look at ObjectVsBroadPhaseLayerFilterTable 
// or ObjectVsBroadPhaseLayerFilterMask for a simpler interface.
class obj_vs_bp_layer_impl_t : public JPH::ObjectVsBroadPhaseLayerFilter
{
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override;
};

// // A contact listener gets notified when bodies (are about to) collide, and when they separate again.
// // Note that this is called from a job so whatever you do here needs to be thread safe.
// // Registering one is entirely optional.
// // An example contact listener
// class MyContactListener : public ContactListener
// {
// public:
//     // See: ContactListener
//     virtual ValidateResult OnContactValidate(const Body &inBody1, const Body &inBody2, RVec3Arg inBaseOffset, const CollideShapeResult &inCollisionResult) override
//     {
//         cout << "Contact validate callback" << endl;

//         // Allows you to ignore a contact before it is created (using layers to not make objects collide is cheaper!)
//         return ValidateResult::AcceptAllContactsForThisBodyPair;
//     }

//     virtual void OnContactAdded(const Body &inBody1, const Body &inBody2, const ContactManifold &inManifold, ContactSettings &ioSettings) override
//     {
//         cout << "A contact was added" << endl;
//     }

//     virtual void OnContactPersisted(const Body &inBody1, const Body &inBody2, const ContactManifold &inManifold, ContactSettings &ioSettings) override
//     {
//         cout << "A contact was persisted" << endl;
//     }

//     virtual void OnContactRemoved(const SubShapeIDPair &inSubShapePair) override
//     {
//         cout << "A contact was removed" << endl;
//     }
// };

// // This class receives callbacks when a virtual character hits something.
// class MyVirtualCharacterContactListener : public CharacterContactListener
// {

// };

// // A body activation listener gets notified when bodies activate and go to sleep
// // Note that this is called from a job so whatever you do here needs to be thread safe.
// // Registering one is entirely optional.
// // An example activation listener
// class MyBodyActivationListener : public BodyActivationListener
// {
// public:
//     virtual void OnBodyActivated(const BodyID &inBodyID, uint64 inBodyUserData) override
//     {
//         cout << "A body got activated" << endl;
//     }

//     virtual void OnBodyDeactivated(const BodyID &inBodyID, uint64 inBodyUserData) override
//     {
//         cout << "A body went to sleep" << endl;
//     }
// };
    
/** Allows a CharacterVirtual to check collision with other CharacterVirtual instances.
    Since CharacterVirtual instances are not registered anywhere, it is up to the 
    application to test collision against relevant characters. The characters could be
    stored in a tree structure to make this more efficient. */
struct game_char_vs_char_handler_t : public JPH::CharacterVsCharacterCollision
{
    /// Add a character to the list of characters to check collision against.
    void Add(JPH::CharacterVirtual *InCharacter) { Characters.push_back(InCharacter); }

    /// Remove a character from the list of characters to check collision against.
    void Remove(const JPH::CharacterVirtual *InCharacter);

    /// Collide a character against other CharacterVirtuals.
    /// @param inCharacter The character to collide.
    /// @param inCenterOfMassTransform Center of mass transform for this character.
    /// @param inCollideShapeSettings Settings for the collision check.
    /// @param inBaseOffset All hit results will be returned relative to this offset, can be zero to get results in world position, but when you're testing far from the origin you get better precision by picking a position that's closer e.g. GetPosition() since floats are most accurate near the origin
    /// @param ioCollector Collision collector that receives the collision results.
    virtual void CollideCharacter(const JPH::CharacterVirtual *inCharacter, JPH::RMat44Arg inCenterOfMassTransform, const JPH::CollideShapeSettings &inCollideShapeSettings, JPH::RVec3Arg inBaseOffset, JPH::CollideShapeCollector &ioCollector) const override;

    /// Cast a character against other CharacterVirtuals.
    /// @param inCharacter The character to cast.
    /// @param inCenterOfMassTransform Center of mass transform for this character.
    /// @param inDirection Direction and length to cast in.
    /// @param inShapeCastSettings Settings for the shape cast.
    /// @param inBaseOffset All hit results will be returned relative to this offset, can be zero to get results in world position, but when you're testing far from the origin you get better precision by picking a position that's closer e.g. GetPosition() since floats are most accurate near the origin
    /// @param ioCollector Collision collector that receives the collision results.
    virtual void CastCharacter(const JPH::CharacterVirtual *inCharacter, JPH::RMat44Arg inCenterOfMassTransform, JPH::Vec3Arg inDirection, const JPH::ShapeCastSettings &inShapeCastSettings, JPH::RVec3Arg inBaseOffset, JPH::CastShapeCollector &ioCollector) const override;

    std::vector<JPH::CharacterVirtual*> Characters;
};

/*
Note that the physics simulation works best if you use SI units (meters, radians, seconds, kg). In 
order for the simulation to be accurate, dynamic objects should be in the order [0.1, 10] meters long, 
have speeds in the order of [0, 500] m/s and have gravity in the order of [0, 10] m/s^2. Static object 
should be in the order [0.1, 2000] meter long. If you are using different units, consider scaling the 
objects before passing them on to the physics simulation.
*/
struct physics_t
{
    void Initialize();
    void Destroy();

    void Tick();

public:
    JPH::PhysicsSystem *PhysicsSystem = nullptr;
    JPH::BodyInterface *BodyInterface = nullptr;

    obj_layer_pair_filter_t ObjectVsObjectFilter; // filters object vs object layers
    bp_layer_interface_t    BroadPhaseLayerInterface; // mapping from object layer to broadphase layer
    obj_vs_bp_layer_impl_t  ObjectVsBroadphaseFilter; // filters object vs broadphase layers

    // List of active virtual characters in the scene so they can collide
    game_char_vs_char_handler_t CharacterVirtualsHandler;

    JPH::TempAllocatorImpl *TempAllocator;
};

// when converting units, differentiate position (convert units) and direction (stay normalized)
JPH::RVec3 ToJoltVec3(vec3 GMathVec3);
vec3 FromJoltVec3(JPH::RVec3 JoltVec3);
JPH::Quat ToJoltQuat(quat GMathQuat);
quat FromJoltQuat(JPH::Quat JoltQuat);
