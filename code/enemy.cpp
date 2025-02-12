
dynamic_array<enemy_t> Enemies;

void enemy_t::Init()
{
    SmoothPath.setlen(MAX_SMOOTH);
    SmoothPathCount = 0;
    SmoothPathIter = 1;
    TimeSinceLastPathFind = 0.f;
}

void enemy_t::Destroy()
{
    SmoothPath.free();
}

void UpdateAllEnemies()
{
    for (size_t i = 0; i < Enemies.lenu(); ++i)
    {
        enemy_t& Enemy = Enemies[i];

        Enemy.TimeSinceLastPathFind += DeltaTime;

        if (Enemy.TimeSinceLastPathFind > 1.f)
        {
            FindSmoothPathTo(Enemy.Position, Player.Root, Enemy.SmoothPath.data, &Enemy.SmoothPathCount);
            Enemy.TimeSinceLastPathFind = 0.f;
            Enemy.SmoothPathIter = 1;
        }

        if (Enemy.SmoothPathIter < Enemy.SmoothPathCount)
        {
            vec3 SteerPoint = *(vec3*)&Enemy.SmoothPath[Enemy.SmoothPathIter*3];
            SupportRenderer.DrawSolidDisc(SteerPoint + vec3(0.f,0.3f,0.f), GM_UP_VECTOR, 8.f);
            vec3 DirToSteerPoint = Normalize(SteerPoint - Enemy.Position);
            vec3 EnemyMoveDelta = DirToSteerPoint * 64.f * DeltaTime;
            float DistTravelled = Magnitude(EnemyMoveDelta);
            float DistToSteerPoint = Magnitude(SteerPoint - Enemy.Position);
            if (DistTravelled >= DistToSteerPoint)
            {
                Enemy.Position = SteerPoint;
                ++Enemy.SmoothPathIter;
            }
            else
            {
                Enemy.Position += EnemyMoveDelta;
            }
        }
    }
}