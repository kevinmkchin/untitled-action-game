#include "physics_bodies.h"

JPH::ObjectLayerPairFilterTable *CreateAndSetupObjectLayers()
{
    JPH::ObjectLayerPairFilterTable *ObjectLayerFilter = 
        new JPH::ObjectLayerPairFilterTable(Layers::NUM_LAYERS);

    ObjectLayerFilter->EnableCollision(Layers::MOVING, Layers::NON_MOVING);
    ObjectLayerFilter->EnableCollision(Layers::MOVING, Layers::MOVING);
    ObjectLayerFilter->EnableCollision(Layers::MOVING, Layers::SENSOR);
    ObjectLayerFilter->EnableCollision(Layers::NON_MOVING, Layers::MOVING); // Non moving only collides with moving
    ObjectLayerFilter->EnableCollision(Layers::SENSOR, Layers::MOVING);

    return ObjectLayerFilter;
}

JPH::BroadPhaseLayerInterfaceTable *CreateAndSetupBroadPhaseLayers()
{
    JPH::BroadPhaseLayerInterfaceTable *MappingTable =
        new JPH::BroadPhaseLayerInterfaceTable(Layers::NUM_LAYERS, BroadPhaseLayers::NUM_LAYERS);

    MappingTable->MapObjectToBroadPhaseLayer(Layers::NON_MOVING, BroadPhaseLayers::NON_MOVING);
    MappingTable->MapObjectToBroadPhaseLayer(Layers::MOVING, BroadPhaseLayers::MOVING);
    MappingTable->MapObjectToBroadPhaseLayer(Layers::SENSOR, BroadPhaseLayers::SENSOR);

    return MappingTable;
}

JPH::ObjectVsBroadPhaseLayerFilterTable *CreateAndSetupObjectVsBroadPhaseFilter(
    JPH::BroadPhaseLayerInterfaceTable *MappingTable,
    JPH::ObjectLayerPairFilterTable *ObjectLayerFilter)
{
    // Auto set up because if the object layers collide then so should the object and broadphase layer
    JPH::ObjectVsBroadPhaseLayerFilterTable *ObjVsBpFilter = 
        new JPH::ObjectVsBroadPhaseLayerFilterTable(*MappingTable, BroadPhaseLayers::NUM_LAYERS,
            *ObjectLayerFilter, Layers::NUM_LAYERS);

    return ObjVsBpFilter;
}

void game_char_vs_char_handler_t::Remove(const JPH::CharacterVirtual *InCharacter)
{
    auto Iter = std::find(Characters.begin(), Characters.end(), InCharacter);
    if (Iter != Characters.end())
        Characters.erase(Iter);
}

void game_char_vs_char_handler_t::CollideCharacter(const JPH::CharacterVirtual *inCharacter, JPH::RMat44Arg inCenterOfMassTransform, const JPH::CollideShapeSettings &inCollideShapeSettings, JPH::RVec3Arg inBaseOffset, JPH::CollideShapeCollector &ioCollector) const
{
    // // Make shape 1 relative to inBaseOffset
    // JPH::Mat44 transform1 = inCenterOfMassTransform.PostTranslated(-inBaseOffset).ToMat44();

    // const JPH::Shape *shape = inCharacter->GetShape();
    // JPH::CollideShapeSettings settings = inCollideShapeSettings;

    // // Iterate over all characters
    // for (const JPH::CharacterVirtual *c : Characters)
    //     if (c != inCharacter
    //         && !ioCollector.ShouldEarlyOut())
    //     {
    //         // Collector needs to know which character we're colliding with
    //         ioCollector.SetUserData(reinterpret_cast<u64>(c));

    //         // Make shape 2 relative to inBaseOffset
    //         JPH::Mat44 transform2 = c->GetCenterOfMassTransform().PostTranslated(-inBaseOffset).ToMat44();

    //         // We need to add the padding of character 2 so that we will detect collision with its outer shell
    //         settings.mMaxSeparationDistance = inCollideShapeSettings.mMaxSeparationDistance + c->GetCharacterPadding();

    //         // Note that this collides against the character's shape without padding, this will be corrected for in CharacterVirtual::GetContactsAtPosition
    //         JPH::CollisionDispatch::sCollideShapeVsShape(shape, c->GetShape(), JPH::Vec3::sReplicate(1.0f), JPH::Vec3::sReplicate(1.0f), transform1, transform2, JPH::SubShapeIDCreator(), JPH::SubShapeIDCreator(), settings, ioCollector);
    //     }

    // // Reset the user data
    // ioCollector.SetUserData(0);
}

void game_char_vs_char_handler_t::CastCharacter(const JPH::CharacterVirtual *inCharacter, JPH::RMat44Arg inCenterOfMassTransform, JPH::Vec3Arg inDirection, const JPH::ShapeCastSettings &inShapeCastSettings, JPH::RVec3Arg inBaseOffset, JPH::CastShapeCollector &ioCollector) const
{
    // // Convert shape cast relative to inBaseOffset
    // JPH::Mat44 transform1 = inCenterOfMassTransform.PostTranslated(-inBaseOffset).ToMat44();
    // JPH::ShapeCast shape_cast(inCharacter->GetShape(), JPH::Vec3::sReplicate(1.0f), transform1, inDirection);

    // // Iterate over all characters
    // for (const JPH::CharacterVirtual *c : Characters)
    //     if (c != inCharacter
    //         && !ioCollector.ShouldEarlyOut())
    //     {
    //         // Collector needs to know which character we're colliding with
    //         ioCollector.SetUserData(reinterpret_cast<u64>(c));

    //         // Make shape 2 relative to inBaseOffset
    //         JPH::Mat44 transform2 = c->GetCenterOfMassTransform().PostTranslated(-inBaseOffset).ToMat44();

    //         // Note that this collides against the character's shape without padding, this will be corrected for in CharacterVirtual::GetFirstContactForSweep
    //         JPH::CollisionDispatch::sCastShapeVsShapeWorldSpace(shape_cast, inShapeCastSettings, c->GetShape(), JPH::Vec3::sReplicate(1.0f), { }, transform2, JPH::SubShapeIDCreator(), JPH::SubShapeIDCreator(), ioCollector);
    //     }

    // // Reset the user data
    // ioCollector.SetUserData(0);
}
