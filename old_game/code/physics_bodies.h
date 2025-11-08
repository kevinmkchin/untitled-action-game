#pragma once

#include "common.h"

namespace Layers
{
    static constexpr JPH::ObjectLayer STATIC = 0;
    static constexpr JPH::ObjectLayer PLAYER = 1;
    static constexpr JPH::ObjectLayer ENEMY = 2;
    static constexpr JPH::ObjectLayer PROJECTILE = 3;
    static constexpr JPH::ObjectLayer GIB = 4;
    static constexpr JPH::ObjectLayer SENSOR = 5; // TRIGGERBOXES
    static constexpr JPH::ObjectLayer NUM_LAYERS = 6;
};

// Each broadphase layer results in a separate bounding volume tree in 
// the broad phase. Too many will be slow.
namespace BroadPhaseLayers
{
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr JPH::BroadPhaseLayer SENSOR(2);
    static constexpr unsigned int NUM_LAYERS(3);
};

// Define the collision rules for the game
JPH::ObjectLayerPairFilterTable *CreateAndSetupObjectLayers();
JPH::BroadPhaseLayerInterfaceTable *CreateAndSetupBroadPhaseLayers();
JPH::ObjectVsBroadPhaseLayerFilterTable *CreateAndSetupObjectVsBroadPhaseFilter(
    JPH::BroadPhaseLayerInterfaceTable *MappingTable,
    JPH::ObjectLayerPairFilterTable *ObjectLayerFilter);


class MyContactListener : public JPH::ContactListener
{
    // Must be thread safe
public:
    virtual JPH::ValidateResult OnContactValidate(const JPH::Body &inBody1, const JPH::Body &inBody2, JPH::RVec3Arg inBaseOffset, const JPH::CollideShapeResult &inCollisionResult) override;
    virtual void OnContactAdded(const JPH::Body &inBody1, const JPH::Body &inBody2, const JPH::ContactManifold &inManifold, JPH::ContactSettings &ioSettings) override;
    virtual void OnContactPersisted(const JPH::Body &inBody1, const JPH::Body &inBody2, const JPH::ContactManifold &inManifold, JPH::ContactSettings &ioSettings) override;
    virtual void OnContactRemoved(const JPH::SubShapeIDPair &inSubShapePair) override;

public:
    // For storing projectile hit infos
    JPH::Mutex ProjectileHitMutex;
};

/*
// A body activation listener gets notified when bodies activate and go to sleep
// Note that this is called from a job so whatever you do here needs to be thread safe.
// Registering one is entirely optional.
// An example activation listener
class MyBodyActivationListener : public BodyActivationListener
{
public:
    virtual void OnBodyActivated(const BodyID &inBodyID, uint64 inBodyUserData) override
    {
        cout << "A body got activated" << endl;
    }

    virtual void OnBodyDeactivated(const BodyID &inBodyID, uint64 inBodyUserData) override
    {
        cout << "A body went to sleep" << endl;
    }
};
*/

// This class receives callbacks when a virtual character hits something.
// Does not need to be thread safe!
class MyVirtualCharacterContactListener : public JPH::CharacterContactListener
{
public:
    struct game_state *GameState = nullptr;

    /// Checks if a character can collide with specified body. Return true if the contact is valid.
    virtual bool OnContactValidate(const JPH::CharacterVirtual *inCharacter, 
        const JPH::BodyID &inBodyID2, const JPH::SubShapeID &inSubShapeID2) override;

    /// Called whenever the character collides with a body.
    virtual void OnContactAdded(const JPH::CharacterVirtual *inCharacter, const JPH::BodyID &inBodyID2, 
        const JPH::SubShapeID &inSubShapeID2, JPH::RVec3Arg inContactPosition, JPH::Vec3Arg inContactNormal, 
        JPH::CharacterContactSettings &ioSettings) override;

    /// Called whenever the character persists colliding with a body.
    virtual void OnContactPersisted(const JPH::CharacterVirtual *inCharacter, const JPH::BodyID &inBodyID2, 
        const JPH::SubShapeID &inSubShapeID2, JPH::RVec3Arg inContactPosition, JPH::Vec3Arg inContactNormal, 
        JPH::CharacterContactSettings &ioSettings) override;

    // OnAdjustBodyVelocity
    // OnCharacterContactValidate
    // OnCharacterContactAdded
    // OnCharacterContactPersisted
    // OnContactSolve
    // OnCharacterContactSolve
};
    
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
