
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

    // TODO (Kevin): I THINK I should replace this with physics based movement so that the enemies 
    //               don't overlap, and also it'll ensure enemy stays on navmesh rather than clipping
    for (size_t i = 0; i < Enemies.lenu(); ++i)
    {
        enemy_t& Enemy = Enemies[i];

        Enemy.TimeSinceLastPathFind += DeltaTime;

        if (Enemy.TimeSinceLastPathFind > 0.3f)
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

            vec3 FlatDir = Normalize(vec3(DirToSteerPoint.x, 0.f, DirToSteerPoint.z));
            Enemy.Orientation = DirectionToOrientation(FlatDir);
        }
    }
}