#pragma once

// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

// The Jolt headers don't include Jolt.h. Always include Jolt.h before including any other Jolt header.
// You can use Jolt.h in your precompiled header to speed up compilation.
#if INTERNAL_BUILD
#define JPH_DEBUG_RENDERER
#endif // INTERNAL_BUILD
// #define JPH_DISABLE_CUSTOM_ALLOCATOR
#include <Jolt/Jolt.h>
// Jolt includes
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/Mutex.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionDispatch.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterTable.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Character/Character.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
// STL includes
#include <iostream>
#include <thread>
#include <cstdarg>

// using body_id = JPH::BodyID;

#include "physics_bodies.h"

void SetJPHMemoryAllocator(MemoryType Allocator);

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

    JPH::TempAllocatorImpl *TempAllocator;

    // List of active virtual characters in the scene so they can collide
    game_char_vs_char_handler_t CharacterVirtualsHandler;

    // Filter to check if two objects can collide based on their ObjectLayer
    JPH::ObjectLayerPairFilterTable *ObjectLayerFilter;
    // Defines a mapping table between object and broadphase layers
    JPH::BroadPhaseLayerInterfaceTable *BroadphaseMapping;
    // Determines if an object layer can collide with a broadphase layer
    JPH::ObjectVsBroadPhaseLayerFilterTable *ObjectVsBroadphaseFilter;

    // ANY DATA THAT CAN BE MUTATED BY CONTACT LISTENER MUST USE CORRESPONDING MUTEX LOCK 
    // Notified when bodies collide and separate
    MyContactListener ContactListener;
};

extern physics_t Physics;

// Converts engine units to and from SI units
inline float ToJoltUnit(float GameUnit);
inline float FromJoltUnit(float JoltUnit);
inline JPH::RVec3 ToJoltVector(vec3 GMathVec3);
inline vec3 FromJoltVector(JPH::RVec3 JoltVec3);
inline JPH::RVec3 ToJoltVectorNoConvert(vec3 GMathVec3); // Directions should stay normalized
inline vec3 FromJoltVectorNoConvert(JPH::RVec3 JoltVec3);
inline JPH::Quat ToJoltQuat(quat GMathQuat);
inline quat FromJoltQuat(JPH::Quat JoltQuat);
