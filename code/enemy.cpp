
// external
global_enemy_state_t EnemySystem;


void global_enemy_state_t::Init()
{
    Enemies = fixed_array<enemy_t>(MaxEnemies, MemoryType::Game);
    Enemies.setlen(MaxEnemies);
    CharacterBodies = fixed_array<JPH::Character *>(MaxCharacterBodies, MemoryType::Game);
    CharacterBodies.setlen(MaxCharacterBodies);

    // TODO are these Shapes and Settings deleted properly? when ref 0
    JPH::RefConst<JPH::Shape> StandingShape = JPH::RotatedTranslatedShapeSettings(
        JPH::Vec3(0, AttackerCapsuleHalfHeightStanding + AttackerCapsuleRadiusStanding, 0), 
        JPH::Quat::sIdentity(), 
        new JPH::CapsuleShape(AttackerCapsuleHalfHeightStanding, AttackerCapsuleRadiusStanding)).Create().Get();

    JPH::Ref<JPH::CharacterSettings> Settings = new JPH::CharacterSettings();
    Settings->mMaxSlopeAngle = JPH::DegreesToRadians(45.0f);
    Settings->mLayer = Layers::ENEMY;
    Settings->mShape = StandingShape;
    Settings->mFriction = 0.5f;
    Settings->mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), -AttackerCapsuleRadiusStanding); // Accept contacts that touch the lower sphere of the capsule

    for (int i = 0; i < MaxCharacterBodies; ++i)
    {
        CharacterBodies[i] = new JPH::Character(
            Settings, ToJoltVector(vec3()), ToJoltQuat(quat()), 0, Physics.PhysicsSystem);

        Physics.BodyInterface->SetUserData(CharacterBodies[i]->GetBodyID(), 
            (u64)BAD_UINDEX);
    }

    for (int i = 0; i < MaxEnemies; ++i)
    {
        Enemies[i].Index = i;
        Enemies[i].Flags = 0x0;
        Enemies[i].Health = 100.f;

        Enemies[i].Position = vec3();
        Enemies[i].Orientation = quat();

        Enemies[i].Animator = nullptr;

        Enemies[i].RigidBody = nullptr;

        Enemies[i].SmoothPath.setlen(MAX_SMOOTH);
        Enemies[i].SmoothPathCount = 0;
        Enemies[i].SmoothPathIter = 1;
        Enemies[i].TimeSinceLastPathFind = 0.f;
    }
}

void global_enemy_state_t::Destroy()
{
    for (int i = 0; i < MaxEnemies; ++i)
    {
        Enemies[i].SmoothPath.free();
    }

    for (int i = 0; i < MaxCharacterBodies; ++i)
    {
        if (Physics.BodyInterface->IsAdded(CharacterBodies[i]->GetBodyID()))
            Physics.BodyInterface->RemoveBody(CharacterBodies[i]->GetBodyID());
        delete CharacterBodies[i]; // ~JPH::Character destroys the body
    }
}

void global_enemy_state_t::RemoveAll()
{
    for (int i = 0; i < MaxEnemies; ++i)
    {
        RemoveEnemy(i);
    }
}

void global_enemy_state_t::SpawnEnemy(game_state *GameState)
{
    enemy_t *NextAvailableEnemy = NULL;
    for (int i = 0; i < MaxEnemies; ++i)
    {
        if (!(Enemies[i].Flags & EnemyFlag_Active))
        {
            NextAvailableEnemy = &Enemies[i];
            break;
        }
    }
    ASSERT(NextAvailableEnemy);

    NextAvailableEnemy->Flags = 0x0;
    NextAvailableEnemy->Flags |= EnemyFlag_Active;
    NextAvailableEnemy->Health = 100.f;

    GetRandomPointOnNavMesh((float*)&NextAvailableEnemy->Position);

    NextAvailableEnemy->RigidBody = NextAvailableCharacterBody();
    if (!NextAvailableEnemy->RigidBody)
    {
        LogError("GAME RUNTIME ERROR: Failed to spawn enemy.");
        RemoveEnemy(NextAvailableEnemy->Index);
        return;
    }
    Physics.BodyInterface->AddBody(NextAvailableEnemy->RigidBody->GetBodyID(), JPH::EActivation::Activate);
    NextAvailableEnemy->RigidBody->SetPosition(ToJoltVector(NextAvailableEnemy->Position));
    // NextAvailableEnemy->RigidBody->SetRotation();
    // THE USER DATA FOR JOLT BODY IS THE ENEMY INDEX FOR NOW
    Physics.BodyInterface->SetUserData(NextAvailableEnemy->RigidBody->GetBodyID(), 
        (u64)NextAvailableEnemy->Index);

    for (size_t i = 0; i < GameState->AnimatorPool.length; ++i)
    {
        if (!GameState->AnimatorPool[i].HasOwner)
        {
            NextAvailableEnemy->Animator = &GameState->AnimatorPool[i];
            NextAvailableEnemy->Animator->HasOwner = true;
            NextAvailableEnemy->Animator->PlayAnimation(
                Assets.Skeleton_Humanoid->Clips[SKE_HUMANOID_RUN], true);
            break;
        }
    }
}

void global_enemy_state_t::RemoveEnemy(u32 EnemyIndex)
{
    Enemies[EnemyIndex].Flags &= ~EnemyFlag_Active;
    Enemies[EnemyIndex].Flags = 0x0;

    RemoveCharacterBodyFromSimulation(Enemies[EnemyIndex].RigidBody);

    Enemies[EnemyIndex].Animator->HasOwner = false;
    Enemies[EnemyIndex].Animator->CurrentAnimation = nullptr;
}

JPH::Character *global_enemy_state_t::NextAvailableCharacterBody()
{
    for (int i = 0; i < MaxCharacterBodies; ++i)
    {
        u32 OwnerIndex = (u32)Physics.BodyInterface->GetUserData(CharacterBodies[i]->GetBodyID());
        if (OwnerIndex == BAD_UINDEX)
            return CharacterBodies[i];
    }
    LogError("GAME RUNTIME ERROR: Failed to retrieve next available JPH::Character body"
        "because all of them are in use.");
    return nullptr;
}

void global_enemy_state_t::RemoveCharacterBodyFromSimulation(JPH::Character *CharacterBody)
{
    if (Physics.BodyInterface->IsAdded(CharacterBody->GetBodyID()))
        Physics.BodyInterface->RemoveBody(CharacterBody->GetBodyID());
    Physics.BodyInterface->SetUserData(CharacterBody->GetBodyID(), (u64)BAD_UINDEX);
}

void NonPhysicsTickAllEnemies(game_state *GameState)
{
    for (int i = 0; i < EnemySystem.MaxEnemies; ++i)
    {
        enemy_t& Enemy = EnemySystem.Enemies[i];
        if (!(Enemy.Flags & EnemyFlag_Active))
            continue;

        if (Enemy.Flags & EnemyFlag_Dead)
        {
            Enemy.DeadTimer -= DeltaTime;
            if (Enemy.DeadTimer <= 0.f)
            {
                if (Enemy.RemainAfterDead)
                {
                    ModelGLTF *CorpseModel = &Assets.ModelsTextured[MT_ATTACKER_CORPSE];
                    ++GameState->StaticInstances.length;
                    FillModelInstanceData(GameState,
                        &GameState->StaticInstances[GameState->StaticInstances.length - 1],
                        Enemy.Position, Enemy.Position, Enemy.Orientation, CorpseModel);
                }

                EnemySystem.RemoveEnemy(Enemy.Index);
                continue;
            }
        }
    }
}

void PrePhysicsTickAllEnemies(game_state *GameState)
{
    for (int i = 0; i < EnemySystem.MaxEnemies; ++i)
    {
        enemy_t& Enemy = EnemySystem.Enemies[i];
        if (!(Enemy.Flags & EnemyFlag_Active))
            continue;

        if (Physics.BodyInterface->IsAdded(Enemy.RigidBody->GetBodyID()))
        {
            Enemy.RigidBody->SetPositionAndRotation(
                ToJoltVector(Enemy.Position), ToJoltQuat(Enemy.Orientation));
        }

        if (!DebugEnemyBehaviourActive || Enemy.Flags & EnemyFlag_Dead)
            continue;

        Enemy.TimeSinceLastPathFind += FixedDeltaTime;

        if (Enemy.TimeSinceLastPathFind > 0.3f)
        {
            // TODO(Kevin): Could I get away with straight path instead of smooth path?
            if (FindSmoothPathTo(Enemy.Position, GameState->Player.Root, Enemy.SmoothPath.data, &Enemy.SmoothPathCount))
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

void PostPhysicsTickAllEnemies(game_state *GameState)
{
    for (int i = 0; i < EnemySystem.MaxEnemies; ++i)
    {
        enemy_t& Enemy = EnemySystem.Enemies[i];
        if (!(Enemy.Flags & EnemyFlag_Active))
            continue;
        
        static const float MaxSeparationDistance = 0.05f;
        Enemy.RigidBody->PostSimulation(MaxSeparationDistance);
        Enemy.Position = FromJoltVector(Enemy.RigidBody->GetPosition());
    }
}

void RenderEnemies(game_state *GameState, const mat4 &ProjFromView, const mat4 &ViewFromWorld)
{
    UseShader(Sha_ModelSkinnedLit);
    glEnable(GL_DEPTH_TEST);
    GLBind4f(Sha_ModelSkinnedLit, "MuzzleFlash", 
        GameState->Player.Weapon.MuzzleFlash.x, 
        GameState->Player.Weapon.MuzzleFlash.y, 
        GameState->Player.Weapon.MuzzleFlash.z, 
        GameState->Player.Weapon.MuzzleFlash.w);
    GLBindMatrix4fv(Sha_ModelSkinnedLit, "Projection", 1, ProjFromView.ptr());
    GLBindMatrix4fv(Sha_ModelSkinnedLit, "View", 1, ViewFromWorld.ptr());

    for (int i = 0; i < EnemySystem.MaxEnemies; ++i)
    {
        enemy_t& Enemy = EnemySystem.Enemies[i];
        if (!(Enemy.Flags & EnemyFlag_Active))
            continue;

        // TODO(Kevin): should use centroid instead of root
        BindUniformsForModelLighting(Sha_ModelSkinnedLit, GameState, Enemy.Position);

        mat4 ModelMatrix = TranslationMatrix(Enemy.Position) * 
            RotationMatrix(Enemy.Orientation) * ScaleMatrix(SI_UNITS_TO_GAME_UNITS);

        GLBindMatrix4fv(Sha_ModelSkinnedLit, "Model", 1, ModelMatrix.ptr());

        GLBindMatrix4fv(Sha_ModelSkinnedLit, "FinalBonesMatrices[0]", MAX_BONES, 
            Enemy.Animator->SkinningMatrixPalette[0].ptr());

        for (size_t i = 0; i < Assets.Model_Attacker->Meshes.length; ++i)
        {
            skinned_mesh_t m = Assets.Model_Attacker->Meshes[i];
            GPUTexture t = Assets.Model_Attacker->Textures[i];

            glActiveTexture(GL_TEXTURE0);
            //glBindTexture(GL_TEXTURE_2D, t.id);
            glBindTexture(GL_TEXTURE_2D, Assets.DefaultEditorTexture.gputex.id);

            glBindVertexArray(m.VAO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.IBO);
            glDrawElements(GL_TRIANGLES, m.IndicesCount, GL_UNSIGNED_INT, nullptr);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
        }
    }
}

void DebugDrawEnemyColliders()
{
#ifdef JPH_DEBUG_RENDERER
    if (!JoltDebugDrawer)
    {
        LogWarning("Called DebugDrawEnemyColliders but JoltDebugDrawer is null.");
        return;
    }

    for (int i = 0; i < EnemySystem.MaxEnemies; ++i)
    {
        enemy_t &Enemy = EnemySystem.Enemies[i];

        if (!Enemy.RigidBody || !Physics.BodyInterface->IsAdded(Enemy.RigidBody->GetBodyID()))
            continue;

        JPH::RMat44 COM = Physics.BodyInterface->GetCenterOfMassTransform(Enemy.RigidBody->GetBodyID());
        const JPH::Shape *EnemyBodyShape = Enemy.RigidBody->GetShape();

        EnemyBodyShape->Draw(JoltDebugDrawer, COM, JPH::Vec3::sReplicate(1.0f), JPH::Color::sRed, false, true);

        // JoltDebugDrawCharacterState(JoltDebugDraw, mCharacter,   
        //     WorldTransform, mCharacter->GetLinearVelocity());
    }
#endif // JPH_DEBUG_RENDERER
}

void HurtEnemy(game_state *GameState, u32 EnemyIndex, float Damage)
{
    enemy_t &Target = EnemySystem.Enemies[EnemyIndex];

    Target.Health -= Damage;

    if (Target.Health <= 0.f && !(Target.Flags & EnemyFlag_Dead))
    {
        KillEnemy(GameState, EnemyIndex);
    }
}

void KillEnemy(game_state *GameState, u32 EnemyIndex)
{
    ++GameState->KillEnemyCounter;

    enemy_t &Target = EnemySystem.Enemies[EnemyIndex];

    Target.Flags |= EnemyFlag_Dead;

    // float f = ENEMYRNG.frand01();
    // LogMessage("%f",f);
    if (ENEMYRNG.frand01() < 0.75f)
    {
        Target.DeadTimer = 1.f;
        Target.RemainAfterDead = true;
        Target.Animator->PlayAnimation(Assets.Skeleton_Humanoid->Clips[SKE_HUMANOID_DEATH], false);
    }
    else
    {
        Target.DeadTimer = 0.f;
        Target.RemainAfterDead = false;

        // maybe randomize num of gibs
        for (int i = 0; i < 4; ++i)
        {
            // actually the directions in which the gibs explode is really
            // important to how they feel. some of the explosions feel much nicer
            // to look at than others currently. maybe they should all sort of arc
            // upwards then shower down
            vec3 GibDirection = RNG.Direction();
            GibDirection.y = fabsf(GibDirection.y)*2.f;
            GibDirection = Normalize(GibDirection);
            SpawnProjectile(projectile_type_enum(PROJECTILE_GENERIC_GIB_0 + RNG.NextInt(0, 1)),
                FromJoltVector(Target.RigidBody->GetCenterOfMassPosition()), 
                vec3(),
                quat(),
                GibDirection * (540.f + RNG.frand() * 100.f),
                RNG.Direction() * 5.0f);
        }
    }

    EnemySystem.RemoveCharacterBodyFromSimulation(Target.RigidBody);
}

