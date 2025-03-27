#include "physics_bodies.h"

JPH::ObjectLayerPairFilterTable *CreateAndSetupObjectLayers()
{
    JPH::ObjectLayerPairFilterTable *ObjectLayerFilter = 
        new JPH::ObjectLayerPairFilterTable(Layers::NUM_LAYERS);

    ObjectLayerFilter->EnableCollision(Layers::STATIC, Layers::PLAYER);
    ObjectLayerFilter->EnableCollision(Layers::STATIC, Layers::ENEMY);
    ObjectLayerFilter->EnableCollision(Layers::STATIC, Layers::PROJECTILE);

    ObjectLayerFilter->EnableCollision(Layers::PLAYER, Layers::STATIC);
    ObjectLayerFilter->EnableCollision(Layers::PLAYER, Layers::ENEMY);
    ObjectLayerFilter->EnableCollision(Layers::PLAYER, Layers::SENSOR);

    ObjectLayerFilter->EnableCollision(Layers::ENEMY, Layers::STATIC);
    ObjectLayerFilter->EnableCollision(Layers::ENEMY, Layers::ENEMY);
    ObjectLayerFilter->EnableCollision(Layers::ENEMY, Layers::PROJECTILE);
    ObjectLayerFilter->EnableCollision(Layers::ENEMY, Layers::SENSOR);

    ObjectLayerFilter->EnableCollision(Layers::PROJECTILE, Layers::STATIC);
    ObjectLayerFilter->EnableCollision(Layers::PROJECTILE, Layers::ENEMY);

    ObjectLayerFilter->EnableCollision(Layers::GIB, Layers::STATIC);

    ObjectLayerFilter->EnableCollision(Layers::SENSOR, Layers::PLAYER);
    ObjectLayerFilter->EnableCollision(Layers::SENSOR, Layers::ENEMY);

    return ObjectLayerFilter;
}

JPH::BroadPhaseLayerInterfaceTable *CreateAndSetupBroadPhaseLayers()
{
    JPH::BroadPhaseLayerInterfaceTable *MappingTable =
        new JPH::BroadPhaseLayerInterfaceTable(Layers::NUM_LAYERS, BroadPhaseLayers::NUM_LAYERS);

    MappingTable->MapObjectToBroadPhaseLayer(Layers::STATIC, BroadPhaseLayers::NON_MOVING);
    MappingTable->MapObjectToBroadPhaseLayer(Layers::PLAYER, BroadPhaseLayers::MOVING);
    MappingTable->MapObjectToBroadPhaseLayer(Layers::ENEMY, BroadPhaseLayers::MOVING);
    MappingTable->MapObjectToBroadPhaseLayer(Layers::PROJECTILE, BroadPhaseLayers::MOVING);
    MappingTable->MapObjectToBroadPhaseLayer(Layers::GIB, BroadPhaseLayers::MOVING);
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


JPH::ValidateResult MyContactListener::OnContactValidate(const JPH::Body &inBody1, const JPH::Body &inBody2, JPH::RVec3Arg inBaseOffset, const JPH::CollideShapeResult &inCollisionResult)
{
    // Allows you to ignore a contact before it is created, but 
    // using layers to not make objects collide is cheaper!
    return JPH::ValidateResult::AcceptAllContactsForThisBodyPair;
}

void MyContactListener::OnContactAdded(const JPH::Body &inBody1, const JPH::Body &inBody2, 
    const JPH::ContactManifold &inManifold, JPH::ContactSettings &ioSettings)
{

    if (inBody1.GetObjectLayer() == Layers::PROJECTILE ||
        inBody2.GetObjectLayer() == Layers::PROJECTILE)
    {
        std::lock_guard Lock(ProjectileHitMutex);

        projectile_hit_info_t PrjHitInfo;
        PrjHitInfo.Body1 = &inBody1;
        PrjHitInfo.Body2 = &inBody2;
        PrjHitInfo.Manifold = &inManifold;
        ProjectileHitInfos.put(PrjHitInfo);

        // NOTE(Kevin): I can set the mass scales of each body to adjust
        //              how far one body knocks back the other. Or I can
        //              set mIsSensor to true for no collision response.
        //              For enemies, their velocities are set every frame
        //              so this doesn't affect them.
        // ioSettings.mInvMassScale1 = 0.f;
    }

}

void MyContactListener::OnContactPersisted(const JPH::Body &inBody1, const JPH::Body &inBody2, 
    const JPH::ContactManifold &inManifold, JPH::ContactSettings &ioSettings)
{

}

void MyContactListener::OnContactRemoved(const JPH::SubShapeIDPair &inSubShapePair)
{

}

/// Checks if a character can collide with specified body. Return true if the contact is valid.
bool MyVirtualCharacterContactListener::OnContactValidate(
    const JPH::CharacterVirtual *inCharacter,
    const JPH::BodyID &inBodyID2,
    const JPH::SubShapeID &inSubShapeID2) 
{
    return true;
}

/// Called whenever the character collides with a body.
/// @param inCharacter Character that is being solved
/// @param inBodyID2 Body ID of body that is being hit
/// @param inSubShapeID2 Sub shape ID of shape that is being hit
/// @param inContactPosition World space contact position
/// @param inContactNormal World space contact normal
/// @param ioSettings Settings returned by the contact callback to indicate 
//         how the character should behave
void MyVirtualCharacterContactListener::OnContactAdded(
    const JPH::CharacterVirtual *inCharacter,
    const JPH::BodyID &inBodyID2,
    const JPH::SubShapeID &inSubShapeID2,
    JPH::RVec3Arg inContactPosition,
    JPH::Vec3Arg inContactNormal,
    JPH::CharacterContactSettings &ioSettings)
{
    if (Physics.BodyInterface->GetObjectLayer(inBodyID2) == Layers::ENEMY)
    {
        Player.Health -= 5.f;
    }
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
