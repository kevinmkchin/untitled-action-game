#pragma once


namespace Layers
{
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING = 1;
    static constexpr JPH::ObjectLayer SENSOR = 2; // TRIGGERBOXES
    static constexpr JPH::ObjectLayer NUM_LAYERS = 3;
};

// Each broadphase layer results in a separate bounding volume tree in the broad phase.
namespace BroadPhaseLayers
{
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr JPH::BroadPhaseLayer SENSOR(2);
    static constexpr unsigned int NUM_LAYERS(3);
};

JPH::ObjectLayerPairFilterTable *CreateAndSetupObjectLayers();
JPH::BroadPhaseLayerInterfaceTable *CreateAndSetupBroadPhaseLayers();
JPH::ObjectVsBroadPhaseLayerFilterTable *CreateAndSetupObjectVsBroadPhaseFilter(
    JPH::BroadPhaseLayerInterfaceTable *MappingTable,
    JPH::ObjectLayerPairFilterTable *ObjectLayerFilter);



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
