
dynamic_array<enemy_t> Enemies;

void enemy_t::Init()
{
    SmoothPath.setlen(MAX_SMOOTH);
    SmoothPathCount = 0;
    SmoothPathIter = 1;
    TimeSinceLastPathFind = 0.f;

    AddToPhysicsSystem();
}

void enemy_t::Destroy()
{
    SmoothPath.free();

    RemoveFromPhysicsSystem();
}

void enemy_t::AddToPhysicsSystem()
{
    // In SI units
    static constexpr float CharacterHeightStanding = 1.5f;
    static constexpr float CharacterRadiusStanding = 0.25f;

    JPH::RefConst<JPH::Shape> StandingShape = JPH::RotatedTranslatedShapeSettings(
        JPH::Vec3(0, 0.5f * CharacterHeightStanding + CharacterRadiusStanding, 0), 
        JPH::Quat::sIdentity(), 
        new JPH::CapsuleShape(0.5f * CharacterHeightStanding, CharacterRadiusStanding)).Create().Get();

    JPH::Ref<JPH::CharacterSettings> Settings = new JPH::CharacterSettings();
    Settings->mMaxSlopeAngle = JPH::DegreesToRadians(45.0f);
    Settings->mLayer = Layers::MOVING;
    Settings->mShape = StandingShape;
    Settings->mFriction = 0.5f;
    Settings->mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -CharacterRadiusStanding); // Accept contacts that touch the lower sphere of the capsule
    
    RigidBody = new JPH::Character(Settings, ToJoltVector(Position), 
        ToJoltQuat(Orientation), 0, Physics.PhysicsSystem);

    RigidBody->AddToPhysicsSystem(JPH::EActivation::Activate);
}

void enemy_t::RemoveFromPhysicsSystem()
{
    RigidBody->RemoveFromPhysicsSystem();
    delete RigidBody;
}

void PrePhysicsTickAllEnemies()
{
    for (size_t i = 0; i < Enemies.lenu(); ++i)
    {
        enemy_t& Enemy = Enemies[i];

        Enemy.TimeSinceLastPathFind += FixedDeltaTime;

        if (Enemy.TimeSinceLastPathFind > 0.3f)
        {
            // TODO replace Player.Root with a target
            if (FindSmoothPathTo(Enemy.Position, Player.Root, Enemy.SmoothPath.data, &Enemy.SmoothPathCount))
            {
                Enemy.TimeSinceLastPathFind = 0.f;
                Enemy.SmoothPathIter = 1;
            }
        }

        if (Enemy.SmoothPathIter < Enemy.SmoothPathCount)
        {
            vec3 SteerPoint = *(vec3*)&Enemy.SmoothPath[Enemy.SmoothPathIter*3];
            vec3 DirToSteerPoint = Normalize(SteerPoint - Enemy.Position);
            vec3 FlatDir = Normalize(vec3(DirToSteerPoint.x, 0.f, DirToSteerPoint.z));
            float DistToSteerPoint = Magnitude(SteerPoint - Enemy.Position);
            if (DistToSteerPoint < 16.f)
            {
                ++Enemy.SmoothPathIter;
            }

            if (Enemy.RigidBody->IsSupported())
            {
                vec3 GroundNormal = FromJoltVectorNoConvert(Enemy.RigidBody->GetGroundNormal());
                vec3 RightDir = Cross(DirToSteerPoint, GroundNormal);
                vec3 ForwardDir = Cross(GroundNormal, RightDir);
                Enemy.RigidBody->SetLinearVelocity(ToJoltVector(ForwardDir * 170.f));
            }

            Enemy.Orientation = DirectionToOrientation(FlatDir);
        }
    }
}

void PostPhysicsTickAllEnemies()
{
    for (size_t i = 0; i < Enemies.lenu(); ++i)
    {
        enemy_t& Enemy = Enemies[i];
        
        static const float MaxSeparationDistance = 0.05f;
        Enemy.RigidBody->PostSimulation(MaxSeparationDistance);
        Enemy.Position = FromJoltVector(Enemy.RigidBody->GetPosition());
    }
}

