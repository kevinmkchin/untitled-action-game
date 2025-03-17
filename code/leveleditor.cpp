
level_editor_t LevelEditor; // extern

std::vector<float> MY_VERTEX_BUFFER;

void level_editor_t::Open()
{
    SelectedTexture = Assets.DefaultEditorTexture;
    IsActive = true;
    EditorCam.Position = vec3(776.0f, 508.9f, 302.7f);
    EditorCam.Rotation = vec3(0.f, 145.2f, -28.8f);
}

void level_editor_t::Close()
{
    IsActive = false;

    // NOTE(Kevin): Should I reset state? or keep data around?
}


float DISC_HANDLE_RADIUS = 10.f;
// These indices are not guaranteed to persist frame to frame. For now (2024-09-04) they index into
// the huge array of all Volumes EDITOR_MAP_VOLUMES. Things will/should move around within the array. 
NiceArray<int, 16> SELECTED_MAP_VOLUMES_INDICES;
std::vector<MapEdit::Vert*> SELECTABLE_VERTICES;
std::vector<MapEdit::Vert*> SELECTED_VERTICES;
std::vector<MapEdit::Face*> SELECTABLE_FACES;


enum class SimpleBrushToolState
{
    NotAvailable,
    NotActive,
    DrawingRectangle,
    DrawingHeight
};
SimpleBrushToolState simpleBrushToolState = SimpleBrushToolState::NotActive;

/// === GRID DEFINITION ===
// I want the grid to be defined by a "UP" and a "RIGHT"
// based on these, I can figure out the orientation of the grid
float GRID_INCREMENT = 32.f;
vec3 GRID_ORIGIN = vec3();
vec3 GRID_UP_VECTOR = GM_UP_VECTOR;
vec3 GRID_RIGHT_VECTOR = GM_RIGHT_VECTOR;

void ResetGridOriginAndOrientation()
{
    GRID_ORIGIN = vec3();
    GRID_UP_VECTOR = GM_UP_VECTOR;
    GRID_RIGHT_VECTOR = GM_RIGHT_VECTOR;
}

mat3 GetGridRotationMatrix()
{
    mat3 rot;
    rot.columns[0] = Normalize(Cross(GRID_UP_VECTOR, GRID_RIGHT_VECTOR)); // forward
    rot.columns[1] = Normalize(GRID_UP_VECTOR); // up
    rot.columns[2] = Normalize(GRID_RIGHT_VECTOR); // right
    return rot;
}

float SnapToGrid(float beforeSnap)
{
    return roundf(beforeSnap / GRID_INCREMENT) * GRID_INCREMENT;
}

vec3 SnapToGrid(vec3 beforeSnap)
{
    vec3 snaptemp = beforeSnap;
    mat3 rot = GetGridRotationMatrix();
    mat3 invRot = rot.GetInverse();
    snaptemp = snaptemp - GRID_ORIGIN;
    snaptemp = invRot * snaptemp;
    snaptemp = vec3(SnapToGrid(snaptemp.x), SnapToGrid(snaptemp.y), SnapToGrid(snaptemp.z));
    snaptemp = rot * snaptemp;
    snaptemp = snaptemp + GRID_ORIGIN;
    return snaptemp;
}

float GetEditorHandleSize(vec3 worldPosition, float sizeInPixels)
{
    float distanceFromCamera = Dot(worldPosition - LevelEditor.EditorCam.Position, LevelEditor.EditorCam.Direction);
    quat camOrientation = EulerToQuat(LevelEditor.EditorCam.Rotation * GM_DEG2RAD);
    vec3 screenPos = WorldPointToScreenPoint(LevelEditor.EditorCam.Position + RotateVector(vec3(distanceFromCamera, 0, 0), camOrientation));
    vec3 screenPos2 = WorldPointToScreenPoint(LevelEditor.EditorCam.Position + RotateVector(vec3(distanceFromCamera, 0, 32.f), camOrientation));
    // scaled by 32 to avoid floating point imprecision
    float screenDist = Magnitude(screenPos - screenPos2);
    return (sizeInPixels*32.f / GM_max(screenDist, 0.0001f));
}

void level_editor_t::Tick()
{
    IsActive = true;
    EnterNextState();

    SDL_SetRelativeMouseMode(MouseCurrent & SDL_BUTTON(SDL_BUTTON_RIGHT) ? SDL_TRUE : SDL_FALSE);

    // EDITOR CAMERA MOVE
    EditorCam.Update(MouseCurrent & SDL_BUTTON(SDL_BUTTON_RIGHT), 0.1f);
    if (MouseCurrent & SDL_BUTTON(SDL_BUTTON_RIGHT))
    {
        float MoveSpeed = 250.f;
        if (KeysCurrent[SDL_SCANCODE_LSHIFT])
            MoveSpeed *= 2.7f;
        EditorCam.DoFlyCamMovement(MoveSpeed);
    }
    ActiveViewMatrix = EditorCam.ViewFromWorldMatrix();

    // DRAW AXIS LINES
    SupportRenderer.DrawLine(vec3(320000.f, 0.f, 0.f), vec3(-320000.f, 0.f, 0.f), vec4(RGB255TO1(205, 56, 9), 0.8f));
    // SupportRenderer.DrawLine(vec3(0.f, 320000.f, 0.f), vec3(0.f, -320000.f, 0.f), vec4(RGB255TO1(67, 123, 9), 0.8f));
    SupportRenderer.DrawLine(vec3(0.f, 0.f, 320000.f), vec3(0.f, 0.f, -320000.f), vec4(RGB255TO1(21, 129, 205), 0.8f));


    DoEditorGUI();

    if (GUI::anyElementHovered || GUI::anyWindowHovered)
    {
        // tools reset? if LMB relased? not always.
        return;
    }

    ResetGridOriginAndOrientation();

    LMBPressedThisFrame = MousePressed & SDL_BUTTON(SDL_BUTTON_LEFT);
    LMBReleasedThisFrame = MouseReleased & SDL_BUTTON(SDL_BUTTON_LEFT);
    LMBIsPressed = MouseCurrent & SDL_BUTTON(SDL_BUTTON_LEFT);


    // === Volume picking ===
    // Triangulation on the fly is efficient but I don't care right now
    // can pick volumes while in most modes
    // can pick nothing to deselect in most modes
    bool volumePickable = VERTEX_MANIP <= ActiveState && ActiveState <= FACE_MANIP;
    if (volumePickable && LMBReleasedThisFrame && HotHandleId == 0)
    {
        u32 pickedVolumeFrameId = PickVolume(LevelEditorVolumes.data, (u32)LevelEditorVolumes.lenu());
        if (pickedVolumeFrameId <= 0)
        {
            SELECTED_MAP_VOLUMES_INDICES.ResetCount();
            ResetFaceToolData();
        }
        else
        {
            if (!KeysCurrent[SDL_SCANCODE_LCTRL])
            {
                SELECTED_MAP_VOLUMES_INDICES.ResetCount();
                ResetFaceToolData();
            }

            int pickedVolumeIndexInMegaArray = pickedVolumeFrameId-1;
            bool exists = false;
            for (int j = 0; j < SELECTED_MAP_VOLUMES_INDICES.count; ++j)
                exists |= SELECTED_MAP_VOLUMES_INDICES.At(j) == pickedVolumeIndexInMegaArray;
            if (!exists && SELECTED_MAP_VOLUMES_INDICES.count < SELECTED_MAP_VOLUMES_INDICES.capacity)
                SELECTED_MAP_VOLUMES_INDICES.PushBack(pickedVolumeIndexInMegaArray);
        }

    }

    // === Selection handles ===
    // on mouse up, disc remains selected
    // on mouse down, if disc not selected, then if ctrl pressed then add to selection but don't start drag
    //                                      if ctrl not pressed then start drag
    // on mouse down, if disc was already selected, then if ctrl pressed then deselect that disc/vert, if ctrl not pressed
    // then start checking for drag
    SELECTABLE_VERTICES.clear();
    SELECTABLE_FACES.clear();
    for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
    {
        MapEdit::Volume& volume = LevelEditorVolumes[SELECTED_MAP_VOLUMES_INDICES.At(i)];

        for (int j = 0; j < volume.verts.lenu(); ++j)
        {
            SELECTABLE_VERTICES.push_back(volume.verts[j]);
        }

        for (int j = 0; j < volume.faces.lenu(); ++j)
        {
            SELECTABLE_FACES.push_back(volume.faces[j]);
        }
    }

    // === State ===
    switch (ActiveState)
    {
        case PLACE_POINT_ENTITY:
            DoPlacePointEntity();
            break;
        case MOVE_POINT_ENTITY:
            DoMovePointEntity();
            break;
        case FACE_MANIP:
            DoFaceManip();
            break;
        case VERTEX_MANIP:
            DoVertexManip();
            break;
        case SIMPLE_BRUSH_TOOL:
            DoSimpleBrushTool();
            break;
    }
}

void level_editor_t::DoEditorGUI()
{
    GUI::BeginWindow(GUI::UIRect(0, 0, RenderTargetGUI.width, 19));
    GUI::EditorBeginHorizontal();
    if (GUI::EditorLabelledButton("SAVE"))
    {
        std::string path = SaveEditableMapFormatDialog();
        if(!path.empty())
        {
            // TODO ...
            if (SaveMap(path.c_str()))
                LogMessage("Saved %s", path.c_str());
            else
                LogError("Failed to save to %s", path.c_str());
            // TODO ...
        }
    }
    if (GUI::EditorLabelledButton("OPEN"))
    {
        std::string path = OpenEditableMapFormatDialog();
        if(!path.empty())
        {
            // TODO ...
            if (LoadMap(path.c_str()))
                LogMessage("Opened %s", path.c_str());
            else
                LogError("Failed to open %s", path.c_str());
            // TODO ...
        }
    }
    if (GUI::EditorLabelledButton("BUILD"))
    {
        std::string path = SaveGameMapDialog();
        if(!path.empty())
        {
            // TODO ...
            if (BuildGameMap(path.c_str()))
                LogMessage("Built %s", path.c_str());
            else
                LogError("Failed to build to %s", path.c_str());
            // TODO ...
        }
    }
    GUI::EditorSpacer(16, 0);

    bool brushToolActive = ActiveState == SIMPLE_BRUSH_TOOL;
    bool vertToolActive = ActiveState == VERTEX_MANIP;
    bool edgeToolActive = ActiveState == EDGE_MANIP;
    bool faceToolActive = ActiveState == FACE_MANIP;
    if (KeysPressed[SDL_SCANCODE_1] || GUI::EditorSelectable_2("BRUSH", &brushToolActive))
    {
        EnterNewStateNextFrame(SIMPLE_BRUSH_TOOL);
    }
    if (KeysPressed[SDL_SCANCODE_2] || GUI::EditorSelectable_2("VERT", &vertToolActive))
    {
        EnterNewStateNextFrame(VERTEX_MANIP);
    }
    if (KeysPressed[SDL_SCANCODE_3] || GUI::EditorSelectable_2("EDGE", &edgeToolActive))
    {
        EnterNewStateNextFrame(EDGE_MANIP);
    }
    if (KeysPressed[SDL_SCANCODE_4] || GUI::EditorSelectable_2("FACE", &faceToolActive))
    {
        EnterNewStateNextFrame(FACE_MANIP);
    }

    GUI::EditorSpacer(16, 0);
    if (GUI::EditorLabelledButton("-", 18))
    {
        GRID_INCREMENT /= 2.f;
    }
    GUI::EditorSpacer(1, 0);
    GUI::EditorText(std::to_string(GRID_INCREMENT).c_str());
    GUI::EditorSpacer(1, 0);
    if (GUI::EditorLabelledButton("+", 18))
    {
        GRID_INCREMENT *= 2.f;
    }

    GUI::EditorSpacer(16, 0);
    GUI::EditorLabelledButton("Toggle Auto Colliders []");
    GUI::EditorLabelledButton("Snap Vertex or Translation to Grid?");

    GUI::EditorEndHorizontal();
    GUI::EndWindow();


    int volc = SELECTED_MAP_VOLUMES_INDICES.count;
    GUI::BeginWindow(GUI::UIRect(RenderTargetGUI.width - 160, 28, 150, 15 + volc * 15));
    GUI::EditorText((std::string("selected volumes (") + std::to_string(volc) + std::string(")")).c_str());
    for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
    {
        int editorMapVolumeIndex = SELECTED_MAP_VOLUMES_INDICES.At(i);
        const MapEdit::Volume& volume = LevelEditorVolumes[editorMapVolumeIndex];
        std::string volPersIdStr = std::to_string(volume.persistId);
        GUI::EditorBeginHorizontal();
        // Gui::EditorLabelledButton("S");
        if (GUI::EditorLabelledButton("X"))
        {
            // TODO remove volume from selection
        }
        GUI::EditorSpacer(4, 0);
        GUI::EditorText(volPersIdStr.c_str());
        GUI::EditorEndHorizontal();
    }
    // Gui::EditorText((std::string("            max ") + std::to_string(SELECTED_MAP_VOLUMES_INDICES.capacity)).c_str());
    GUI::EndWindow();


    GUI::BeginWindow(GUI::UIRect(RenderTargetGUI.width - 210, RenderTargetGUI.height - 310, 200, 300));
    GUI::EditorText("Texture");
    
    static bool ShowTextureBrowser = false;
    if (GUI::EditorImageButton(SelectedTexture.gputex.id, ivec2(160, 160)))
    {
        ShowTextureBrowser = true;
    }

    if (ShowTextureBrowser)
    {
        GUI::BeginWindow(GUI::UIRect(0, 0, RenderTargetGUI.width, RenderTargetGUI.height));
        
        GUI::EditorBeginHorizontal();
        if (GUI::EditorLabelledButton("Close"))
            ShowTextureBrowser = false;
        GUI::EditorSpacer(4, 0);
        if (GUI::EditorLabelledButton("Load Textures"))
        {
            std::vector<std::string> paths = OpenImageFilesDialog();
            for (const auto& fp : paths)
            {
                db_tex_t loadedtex = Assets.LoadNewTexture(fp.c_str());
            }
        }
        GUI::EditorEndHorizontal();

        GUI::EditorSpacer(0, 8);
        GUI::EditorBeginGrid(RenderTargetGUI.width, 8000);
        for (auto& pair : Assets.Textures)
        {
            GUI::EditorBeginGridItem(162, 186);

            db_tex_t t = pair.second;
            if (GUI::EditorImageButton(t.gputex.id, ivec2(160, 160)))
            {
                SelectedTexture = t;
                ShowTextureBrowser = false;
            }
            char displaystr[128] = {0};
            stbsp_sprintf(displaystr, "ID: %d\t\t%dx%d", t.persistId, t.gputex.width, t.gputex.height);
            GUI::EditorText(displaystr);

            GUI::EditorEndGridItem();
        }
        GUI::EditorEndGrid();

        GUI::EndWindow();
    }

    // TODO(Kevin): Have shortcuts for these actions so you can go click click shortcut click click shortcut
    GUI::EditorLabelledButton("Apply Texture to Volume");
    if (SelectedFace != NULL && GUI::EditorLabelledButton("Apply Texture to Face"))
    {
        SelectedFace->texture = SelectedTexture; 
    }
    GUI::EndWindow();

    GUI::BeginWindow(GUI::UIRect(RenderTargetGUI.width - 250, 150, 240, 150));
    bool PlacePointEntityActive = ActiveState == PLACE_POINT_ENTITY;
    GUI::EditorBeginHorizontal();
    if (GUI::EditorSelectable_2("", &PlacePointEntityActive))
    {
        EntityTypeToPlace = POINT_PLAYER_SPAWN;
        EnterNewStateNextFrame(PLACE_POINT_ENTITY);
    }
    GUI::EditorText(" Place POINT_PLAYER_SPAWN");
    GUI::EditorEndHorizontal();
    GUI::EditorBeginHorizontal();
    if (GUI::EditorSelectable_2("", &PlacePointEntityActive))
    {
        EntityTypeToPlace = POINT_LIGHT;
        EnterNewStateNextFrame(PLACE_POINT_ENTITY);
    }
    GUI::EditorText(" Place POINT_LIGHT");
    GUI::EditorEndHorizontal();
    GUI::EditorBeginHorizontal();
    if (GUI::EditorSelectable_2("", &PlacePointEntityActive))
    {
        EntityTypeToPlace = DIRECTIONAL_LIGHT_PROPERTIES;
        EnterNewStateNextFrame(PLACE_POINT_ENTITY);
    }
    GUI::EditorText(" Place DIRECTIONAL_LIGHT_PROPERTIES");
    GUI::EditorEndHorizontal();
    GUI::EndWindow();
}

void level_editor_t::EnterNewStateNextFrame(editor_state_t NextState)
{
    if (ActiveState == NextState) 
        return; // then ignore

    QueuedState = NextState;
}

void level_editor_t::EnterNextState()
{
    // Handle all state switching here!

    if (QueuedState == INVALID_EDITOR_STATE)
        return;

    // Deactivate active state
    switch (ActiveState)
    {
        case SIMPLE_BRUSH_TOOL:
        case VERTEX_MANIP:
        case EDGE_MANIP:
        case FACE_MANIP:
            HotHandleId = 0;
            ResetFaceToolData();
            break;
    }

    LastState = ActiveState;
    ActiveState = QueuedState;
    QueuedState = INVALID_EDITOR_STATE;

    // Activate new active state
    switch (ActiveState)
    {
        case SIMPLE_BRUSH_TOOL:
            SELECTED_MAP_VOLUMES_INDICES.ResetCount();
            break;
    }
}

void level_editor_t::DrawEntityBillboards()
{
    // GUI::PrimitivePanel(GUI::UIRect(0, 0, 512, 512), SupportRenderer.EntityBillboardAtlas.id);
    // TODO also fade out and don't draw gizmos that are small

    for (size_t Index = 0; Index < LevelEntities.lenu(); ++Index)
    {
        const level_entity_t& Ent = LevelEntities[Index];
        SupportRenderer.DoPickableBillboard((u32)Index+1, 
            Ent.Position, -EditorCam.Direction, (int)Ent.Type);
    }
}

void level_editor_t::ResetFaceToolData()
{
    SelectedFace = NULL;
}

u32 level_editor_t::PickVolume(MapEdit::Volume *volumes, u32 arraycount)
{
    float PickableTrianglesBuf[400];

    // returns the 1 + index of the volume in the provided array. 0 is nothing picked.
    for (u32 volumeFrameId = 1; volumeFrameId <= arraycount; ++volumeFrameId)
    {
        vec3 idrgb = SupportRenderer.HandleIdToRGB(volumeFrameId);
        MapEdit::Volume& vol = volumes[volumeFrameId - 1];
        for (size_t i = 0; i < vol.faces.lenu(); ++i)
        {
            MapEdit::Face *face = vol.faces[i];
            int count;
            // TODO(Kevin): instead of triangulate every face every time, probably cache this somehow?
            MapEdit::TriangulateFace_QuickDumb_WithColor(*face, idrgb, PickableTrianglesBuf, &count);
            SupportRenderer.AddTrianglesToPickableHandles(PickableTrianglesBuf, count);
        }
    }
    u32 pickedVolumeFrameId = SupportRenderer.FlushHandles(MousePos, RenderTargetGame, ActiveViewMatrix, ActivePerspectiveMatrix, false);
    return pickedVolumeFrameId;
}

u32 level_editor_t::PickFace(MapEdit::Face **faces, u32 arraycount)
{
    float PickableTrianglesBuf[400];

    // returns 1 + index of the face in the provided array. 0 is nothing picked.
    for (u32 id = 1; id <= arraycount; ++id)
    {
        MapEdit::Face *face = faces[id-1];
        vec3 idrgb = SupportRenderer.HandleIdToRGB(id);
        int count;
        // TODO(Kevin): instead of triangulate every face every time, probably cache this somehow?
        MapEdit::TriangulateFace_QuickDumb_WithColor(*face, idrgb, PickableTrianglesBuf, &count);
        SupportRenderer.AddTrianglesToPickableHandles(PickableTrianglesBuf, count);
    }
    u32 faceId = SupportRenderer.FlushHandles(MousePos, RenderTargetGame, ActiveViewMatrix, ActivePerspectiveMatrix, false);
    return faceId;
}

bool level_editor_t::PickPointAndNormalInLevel(vec3 *PlanePoint, vec3 *PlaneNormal)
{
    u32 pickedVolumeFrameId = PickVolume(LevelEditorVolumes.data, (u32)LevelEditorVolumes.lenu());
    if (pickedVolumeFrameId > 0)
    {
        MapEdit::Volume& vol = LevelEditorVolumes[pickedVolumeFrameId-1];
        dynamic_array<MapEdit::Face*> faces = vol.faces;
        u32 faceIndex = PickFace(faces.data, (u32)faces.lenu());
        MapEdit::Face *drawingFace = faces[faceIndex-1];

        *PlaneNormal = drawingFace->QuickNormal();

        vec3 ws = ScreenPointToWorldPoint(MousePos, 0.f);
        vec3 wr = ScreenPointToWorldRay(MousePos);
        vec3 intersectionPoint;
        IntersectPlaneAndLineWithDirections(drawingFace->loopbase->v->pos, *PlaneNormal, ws, wr, &intersectionPoint);
        
        *PlanePoint = intersectionPoint;
    }
    else
    {
        vec3 ws = ScreenPointToWorldPoint(MousePos, 0.f);
        vec3 wr = ScreenPointToWorldRay(MousePos);
        float f = (0.f - ws.y) / wr.y;
        if (f < 0.f)
            return false;
        simpleBrushToolState = SimpleBrushToolState::DrawingRectangle;
        *PlanePoint = ws + wr * f;
        *PlaneNormal = GM_UP_VECTOR;
    }
    return true;
}

void level_editor_t::DoPlacePointEntity()
{
    if (LMBReleasedThisFrame)
    {
        vec3 PickedPoint, PickedNormal;
        bool ValidPointWasPicked = PickPointAndNormalInLevel(&PickedPoint, &PickedNormal);
        if (!ValidPointWasPicked)
            return;

        level_entity_t PlacedPointEntity;
        PlacedPointEntity.Type = EntityTypeToPlace;
        PlacedPointEntity.Position = SnapToGrid(PickedPoint);
        PlacedPointEntity.Rotation = vec3();
        if (EntityTypeToPlace == DIRECTIONAL_LIGHT_PROPERTIES) // TODO introduce property window
            PlacedPointEntity.Rotation = vec3(-1.0f, 0.9f, -0.16f);
        LevelEntities.put(PlacedPointEntity);

        EnterNewStateNextFrame(MOVE_POINT_ENTITY);
    }
}

void level_editor_t::DoMovePointEntity()
{
    static int HotEntityIndex = -1;
    static vec3 DragPlanePoint;
    static vec3 EntityMoveStartPoint;

    if (LMBPressedThisFrame)
    {
        DrawEntityBillboards();
        u32 PickedId = SupportRenderer.FlushHandles(MousePos, RenderTargetGame, ActiveViewMatrix, ActivePerspectiveMatrix, false);
        if (PickedId == 0) 
            return;
        u32 PickedEntityIndex = PickedId - 1;

        SelectedEntityIndex = PickedEntityIndex;
        HotEntityIndex = PickedEntityIndex;
    }

    if (LMBIsPressed && HotEntityIndex == SelectedEntityIndex && SelectedEntityIndex >= 0)
    {
        level_entity_t& Ent = LevelEntities[SelectedEntityIndex];
        vec3 WorldPosMouse = ScreenPointToWorldPoint(MousePos, 0.f);
        vec3 WorldRayMouse = ScreenPointToWorldRay(MousePos);

        if (LMBPressedThisFrame) // refactor translation code into one util function 
        {
            EntityMoveStartPoint = Ent.Position;
            IntersectPlaneAndLineWithDirections(Ent.Position, -EditorCam.Direction, 
                WorldPosMouse, WorldRayMouse, &DragPlanePoint);
        }
        else
        {
            vec3 TotalTranslation;
            if (KeysCurrent[SDL_SCANCODE_LALT])
            {
                vec3 Intersect;
                IntersectPlaneAndLineWithDirections(DragPlanePoint, 
                    vec3(-EditorCam.Direction.x, 0.f, -EditorCam.Direction.z), 
                    WorldPosMouse, WorldRayMouse, &Intersect);
                float yTranslation = Dot((Intersect - DragPlanePoint), GM_UP_VECTOR);
                TotalTranslation = vec3(0.f,yTranslation,0.f);
                TotalTranslation = SnapToGrid(TotalTranslation);
            }
            else
            {
                vec3 Intersect;
                IntersectPlaneAndLine(DragPlanePoint, GM_UP_VECTOR, 
                    WorldPosMouse, WorldRayMouse, &Intersect);
                TotalTranslation = Intersect - DragPlanePoint;
                TotalTranslation = SnapToGrid(TotalTranslation);
            }

            Ent.Position = EntityMoveStartPoint + TotalTranslation;

            if (KeysCurrent[SDL_SCANCODE_LALT])
            {
                SupportRenderer.DrawLine(EntityMoveStartPoint, Ent.Position, vec4(0,0.8f,0,1));
                GUI::PrimitiveTextFmt(GUI::MouseXInGUI+3,GUI::MouseYInGUI-3,GUI::GetFontSize(),GUI::LEFT,
                    "Y: %f", TotalTranslation.y);
            }
        }
    }

    if (LMBReleasedThisFrame)
    {
        HotEntityIndex = -1;
    }
}

void level_editor_t::DoFaceManip()
{
    static vec3 DragPlanePoint;

    u32 hoveredFaceId = PickFace(SELECTABLE_FACES.data(), (u32)SELECTABLE_FACES.size());

    if (hoveredFaceId > 0)
    {
        MapEdit::Face *face = SELECTABLE_FACES[hoveredFaceId-1];
        face->hovered = true;

        if (LMBPressedThisFrame)
        {
            HotHandleId = hoveredFaceId;
            SelectedFace = face;

            SELECTED_VERTICES = face->GetVertices();

            for (MapEdit::Vert *vert : SELECTED_VERTICES)
                vert->poscache = vert->pos;

            vec3 worldpos_mouse = ScreenPointToWorldPoint(MousePos, 0.f);
            vec3 worldray_mouse = ScreenPointToWorldRay(MousePos);
            IntersectPlaneAndLineWithDirections(face->loopbase->v->pos, face->QuickNormal(), worldpos_mouse, worldray_mouse, &DragPlanePoint);
        }
    }

    if (LMBIsPressed && HotHandleId > 0)
    {
        MapEdit::Face *hotFace = SELECTABLE_FACES[HotHandleId-1];

        vec3 worldpos_mouse = ScreenPointToWorldPoint(MousePos, 0.f);
        vec3 worldray_mouse = ScreenPointToWorldRay(MousePos);

        vec3 TotalTranslation;
        if (KeysCurrent[SDL_SCANCODE_LALT])
        {
            vec3 intersect;
            IntersectPlaneAndLineWithDirections(DragPlanePoint, vec3(-EditorCam.Direction.x, 0.f, -EditorCam.Direction.z), worldpos_mouse, worldray_mouse, &intersect);
            float yTranslation = Dot((intersect - DragPlanePoint), GM_UP_VECTOR);
            TotalTranslation = vec3(0.f,yTranslation,0.f);
            TotalTranslation = SnapToGrid(TotalTranslation);
        }
        else
        {
            vec3 intersect;
            IntersectPlaneAndLine(DragPlanePoint, GM_UP_VECTOR, worldpos_mouse, worldray_mouse, &intersect);
            TotalTranslation = intersect - DragPlanePoint;
            TotalTranslation = SnapToGrid(TotalTranslation);
        }

        if (KeysCurrent[SDL_SCANCODE_ESCAPE])
        {
            for (MapEdit::Vert *vert : SELECTED_VERTICES)
                vert->pos = vert->poscache;
            HotHandleId = 0;
        }
        else
        {
            for (MapEdit::Vert *vert : SELECTED_VERTICES)
            {
                vert->pos = vert->poscache + TotalTranslation;
            }
        }
        for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
        {
            MapEdit::Volume& selectedVol = LevelEditorVolumes[SELECTED_MAP_VOLUMES_INDICES.At(i)];
            for (size_t j = 0; j < selectedVol.faces.lenu(); ++j)
            {
                MapEdit::Face *face = selectedVol.faces[j];
                MY_VERTEX_BUFFER.clear();
                MapEdit::TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                RebindGPUMesh(&face->facemesh, sizeof(float)*MY_VERTEX_BUFFER.size(), MY_VERTEX_BUFFER.data());
            }
        }
    }

    if (LMBReleasedThisFrame && HotHandleId > 0)
    {
        HotHandleId = 0;
    }
}

void level_editor_t::DoVertexManip()
{
    static vec3 DragPlanePoint;
    static bool LCtrlDownOnLeftMouseDown = false;
    static bool AlreadySelectedOnLeftMouseDown = false;
    static vec3 TotalTranslation;

    if (LMBPressedThisFrame)
    {
        LCtrlDownOnLeftMouseDown = KeysCurrent[SDL_SCANCODE_LCTRL];

        for (u32 id = 1; id <= SELECTABLE_VERTICES.size(); ++id)
        {
            MapEdit::Vert *vert = SELECTABLE_VERTICES[id-1];
            SupportRenderer.DoDiscHandle(id, vert->pos, EditorCam.Position, GetEditorHandleSize(vert->pos, DISC_HANDLE_RADIUS + 4.f));
        }
        u32 clickedId = SupportRenderer.FlushHandles(MousePos, RenderTargetGame, ActiveViewMatrix, ActivePerspectiveMatrix, false);

        if (clickedId > 0)
        {
            HotHandleId = clickedId;
            MapEdit::Vert *hotVert = SELECTABLE_VERTICES[HotHandleId-1];
            auto clickedIter = std::find(SELECTED_VERTICES.begin(), SELECTED_VERTICES.end(), hotVert);
            AlreadySelectedOnLeftMouseDown = clickedIter != SELECTED_VERTICES.end();
            if (!AlreadySelectedOnLeftMouseDown)
            {
                if (!LCtrlDownOnLeftMouseDown)
                    SELECTED_VERTICES.clear();
                SELECTED_VERTICES.push_back(hotVert);
                hotVert->poscache = hotVert->pos;
            }

            vec3 worldpos_mouse = ScreenPointToWorldPoint(MousePos, 0.f);
            vec3 worldray_mouse = ScreenPointToWorldRay(MousePos);
            IntersectPlaneAndLineWithDirections(hotVert->pos, -EditorCam.Direction, worldpos_mouse, worldray_mouse, &DragPlanePoint);
        }
        else if (!LCtrlDownOnLeftMouseDown)
        {
            SELECTED_VERTICES.clear();
        }
        else
        {
            // if id == 0 (nothing clicked) then check for click against volumes
            // also draw all the other volumes with a clickable rgb id so we can switch which volume displays their vertices
            // draw the selected volume(s) with rgb(0) so it is ignored when clicked
            // draw these discs on top of all the volumes, but the discs themselves must do depth test so a disc closer to camera
            // is clicked first.
        }
    }

    if (LMBIsPressed && HotHandleId > 0)
    {
        MapEdit::Vert *hotVert = SELECTABLE_VERTICES[HotHandleId-1];

        vec3 worldpos_mouse = ScreenPointToWorldPoint(MousePos, 0.f);
        vec3 worldray_mouse = ScreenPointToWorldRay(MousePos);

        TotalTranslation = vec3();
        if (KeysCurrent[SDL_SCANCODE_LALT])
        {
            vec3 intersect;
            IntersectPlaneAndLineWithDirections(DragPlanePoint, vec3(-EditorCam.Direction.x, 0.f, -EditorCam.Direction.z), worldpos_mouse, worldray_mouse, &intersect);
            float yTranslation = Dot((intersect - DragPlanePoint), GM_UP_VECTOR);
            TotalTranslation = vec3(0.f,yTranslation,0.f);
            TotalTranslation = SnapToGrid(TotalTranslation);
        }
        else
        {
            vec3 intersect;
            IntersectPlaneAndLine(DragPlanePoint, GM_UP_VECTOR, worldpos_mouse, worldray_mouse, &intersect);
            TotalTranslation = intersect - DragPlanePoint;
            TotalTranslation = SnapToGrid(TotalTranslation);
        }

        if (KeysCurrent[SDL_SCANCODE_ESCAPE])
        {
            for (MapEdit::Vert *vert : SELECTED_VERTICES)
                vert->pos = vert->poscache;
            HotHandleId = 0;
            LCtrlDownOnLeftMouseDown = false;
            AlreadySelectedOnLeftMouseDown = false;
        }
        else
        {
            for (MapEdit::Vert *vert : SELECTED_VERTICES)
            {
                vert->pos = vert->poscache + TotalTranslation;
            }
        }
        for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
        {
            MapEdit::Volume& selectedVol = LevelEditorVolumes[SELECTED_MAP_VOLUMES_INDICES.At(i)];
            for (size_t j = 0; j < selectedVol.faces.lenu(); ++j)
            {
                MapEdit::Face *face = selectedVol.faces[j];
                MY_VERTEX_BUFFER.clear();
                MapEdit::TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                RebindGPUMesh(&face->facemesh, sizeof(float)*MY_VERTEX_BUFFER.size(), MY_VERTEX_BUFFER.data());
            }
        }
    }

    if (LMBReleasedThisFrame)
    {
        if (HotHandleId > 0)
        {
            MapEdit::Vert *hotVert = SELECTABLE_VERTICES[HotHandleId-1];
            if (LCtrlDownOnLeftMouseDown)
            {
                auto hotVertIter = std::find(SELECTED_VERTICES.begin(), SELECTED_VERTICES.end(), hotVert);
                if (hotVertIter != SELECTED_VERTICES.end())
                {
                    if (AlreadySelectedOnLeftMouseDown && TotalTranslation != vec3())
                        SELECTED_VERTICES.erase(hotVertIter);
                }
                else
                {
                    LogError("wtf");
                }
            }
            else if (TotalTranslation != vec3())
            {
                SELECTED_VERTICES.clear();
                SELECTED_VERTICES.push_back(hotVert);
            }
        }

        HotHandleId = 0;
        LCtrlDownOnLeftMouseDown = false;
        AlreadySelectedOnLeftMouseDown = false;
    }
}

void level_editor_t::DoSimpleBrushTool()
{
    static vec3 rectstartpoint;
    static vec3 drawingplanenormal;
    static vec3 drawinghorizontal;
    static vec3 drawingvertical;
    static vec3 rectendpoint;
    switch (simpleBrushToolState)
    {
        case SimpleBrushToolState::NotActive:
            if (LMBPressedThisFrame)
            {
                // We pick a point and normal on a plane
                bool ValidPointWasPicked = 
                    PickPointAndNormalInLevel(&rectstartpoint, &drawingplanenormal);
                if (!ValidPointWasPicked)
                    break;

                rectstartpoint = SnapToGrid(rectstartpoint);
                // no matter what, at least one of these vectors is GM_RIGHT_VECTOR rotated around y-axis.
                vec3 flattenedNormal = vec3(drawingplanenormal.x, 0.f, drawingplanenormal.z);
                if (Magnitude(flattenedNormal) < 0.001f)
                    flattenedNormal = GM_FORWARD_VECTOR;
                else if (GM_abs(drawingplanenormal.y) < 0.001f)
                    flattenedNormal = GM_UP_VECTOR;
                else
                    flattenedNormal = Normalize(flattenedNormal);
                drawinghorizontal = Normalize(Cross(drawingplanenormal, flattenedNormal));
                drawingvertical = Normalize(Cross(drawingplanenormal, drawinghorizontal));
                
                simpleBrushToolState = SimpleBrushToolState::DrawingRectangle;
            }
            break;
        case SimpleBrushToolState::DrawingRectangle:
            GRID_ORIGIN = rectstartpoint;
            GRID_RIGHT_VECTOR = drawinghorizontal;
            GRID_UP_VECTOR = drawingplanenormal;
            if (LMBIsPressed)
            {
                vec3 ws = ScreenPointToWorldPoint(MousePos, 0.f);
                vec3 wr = ScreenPointToWorldRay(MousePos);
                vec3 intersection;
                IntersectPlaneAndLineWithDirections(rectstartpoint, drawingplanenormal, ws, wr, &intersection);
                vec3 endpoint = intersection;
                endpoint = SnapToGrid(endpoint);
                vec3 startToEndVector = vec3(endpoint - rectstartpoint);

                // SupportRenderer.DrawSolidDisc(rectstartpoint, drawingplanenormal, 4, vec4(0,0,1,1));
                // SupportRenderer.DrawSolidDisc(endpoint, drawingplanenormal, 4, vec4(0,0,1,1));
                // SupportRenderer.DrawLine(rectstartpoint, rectstartpoint + drawinghorizontal * 16.f, vec4(0,0,1,1), 2.f); 
                // SupportRenderer.DrawLine(rectstartpoint, rectstartpoint + drawingvertical * 16.f, vec4(0,0,1,1), 2.f); 
                vec3 startToEndProjOnToHorizontal = Dot(startToEndVector, drawinghorizontal) * drawinghorizontal;
                vec3 startToEndProjOnToVertical = Dot(startToEndVector, drawingvertical) * drawingvertical;
                SupportRenderer.DrawLine(rectstartpoint, rectstartpoint + startToEndProjOnToVertical,
                                  vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                SupportRenderer.DrawLine(rectstartpoint, rectstartpoint + startToEndProjOnToHorizontal,
                                  vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                SupportRenderer.DrawLine(endpoint, endpoint - startToEndProjOnToVertical,
                                  vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
                SupportRenderer.DrawLine(endpoint, endpoint - startToEndProjOnToHorizontal,
                                  vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
            }
            if (LMBReleasedThisFrame)
            {
                vec3 ws = ScreenPointToWorldPoint(MousePos, 0.f);
                vec3 wr = ScreenPointToWorldRay(MousePos);
                vec3 intersection;
                IntersectPlaneAndLineWithDirections(rectstartpoint, drawingplanenormal, ws, wr, &intersection);
                rectendpoint = intersection;
                rectendpoint = SnapToGrid(rectendpoint);
                vec3 startToEndVector = vec3(rectendpoint - rectstartpoint);
                vec3 startToEndProjOnToHorizontal = Dot(startToEndVector, drawinghorizontal) * drawinghorizontal;
                vec3 startToEndProjOnToVertical = Dot(startToEndVector, drawingvertical) * drawingvertical;
                if (Magnitude(startToEndProjOnToHorizontal)+0.05f >= GRID_INCREMENT && Magnitude(startToEndProjOnToVertical)+0.05f >= GRID_INCREMENT)
                    simpleBrushToolState = SimpleBrushToolState::DrawingHeight;
                else
                    simpleBrushToolState = SimpleBrushToolState::NotActive;
            }
            if (KeysPressed[SDL_SCANCODE_ESCAPE])
                simpleBrushToolState = SimpleBrushToolState::NotActive;

            break;
        case SimpleBrushToolState::DrawingHeight:

            vec3 startToEndVector = vec3(rectendpoint - rectstartpoint);

            float zcomponent = Dot(startToEndVector, drawinghorizontal);
            float xcomponent = Dot(startToEndVector, drawingvertical);
            vec3 startToEndProjOnToHorizontal = zcomponent * drawinghorizontal;
            vec3 startToEndProjOnToVertical = xcomponent * drawingvertical;

            GRID_ORIGIN = rectendpoint;
            GRID_UP_VECTOR = startToEndProjOnToHorizontal;
            GRID_RIGHT_VECTOR = startToEndProjOnToVertical;

            // let plane be defined at point rectendpoint with normal -cameraDirection
            // then the point of intersection between mouse ray and the plane - endrectpoint and project it onto
            // the direction of translation e.g. GM_UP_VECTOR
            vec3 drawingSurfaceNormal = drawingplanenormal;
            static vec3 height = vec3();
            static vec3 heightBeforeSnap = vec3();
            vec3 pn = -EditorCam.Direction;
            vec3 pp = rectendpoint + heightBeforeSnap;
            vec3 wp = ScreenPointToWorldPoint(MousePos, 0.f);
            vec3 wr = ScreenPointToWorldRay(MousePos);
            vec3 intersection;
            IntersectPlaneAndLineWithDirections(pp, pn, wp, wr, &intersection);
            float trueHeightComp = Dot((intersection - rectendpoint), drawingSurfaceNormal);
            float heightcomponent = SnapToGrid(trueHeightComp);
            height = heightcomponent * drawingSurfaceNormal;
            heightBeforeSnap = trueHeightComp * drawingSurfaceNormal;

            vec3 floorPointA = rectstartpoint;
            vec3 floorPointB = rectstartpoint + startToEndProjOnToVertical;
            vec3 floorPointC = rectendpoint;
            vec3 floorPointD = rectendpoint - startToEndProjOnToVertical;
            // given how i've set up points ABCD:
            //      if xcomponent < 0, swap A and B, swap C and D
            //      if zcomponent < 0, swap B and C, swap A and D
            if (xcomponent < 0)
            {
                auto temp = floorPointA;
                floorPointA = floorPointB;
                floorPointB = temp;
                temp = floorPointC;
                floorPointC = floorPointD;
                floorPointD = temp;
            }
            if (zcomponent < 0)
            {
                auto temp = floorPointB;
                floorPointB = floorPointC;
                floorPointC = temp;
                temp = floorPointA;
                floorPointA = floorPointD;
                floorPointD = temp;
            }
            vec3 ceilPointA = floorPointA + height;
            vec3 ceilPointB = floorPointB + height;
            vec3 ceilPointC = floorPointC + height;
            vec3 ceilPointD = floorPointD + height;

            SupportRenderer.DrawLine(floorPointA, floorPointB, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
            SupportRenderer.DrawLine(floorPointA, floorPointD, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
            SupportRenderer.DrawLine(floorPointC, floorPointB, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
            SupportRenderer.DrawLine(floorPointC, floorPointD, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);

            SupportRenderer.DrawLine(floorPointA, ceilPointA, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
            SupportRenderer.DrawLine(floorPointB, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
            SupportRenderer.DrawLine(floorPointC, ceilPointC, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
            SupportRenderer.DrawLine(floorPointD, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);

            SupportRenderer.DrawLine(ceilPointA, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
            SupportRenderer.DrawLine(ceilPointA, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
            SupportRenderer.DrawLine(ceilPointC, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);
            SupportRenderer.DrawLine(ceilPointC, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0), 2.f);

            if (LMBReleasedThisFrame)
            {
                // if heigth component is zero or too small then just default to current grid increment

                MapEdit::Volume createdVolume;
                createdVolume.persistId = MapEdit::FreshVolumePersistId();

                // complete with drawn height
                MapEdit::Vert *fv0 = MapEdit::CreateVert(floorPointA, &createdVolume);
                MapEdit::Vert *fv1 = MapEdit::CreateVert(floorPointB, &createdVolume);
                MapEdit::Vert *fv2 = MapEdit::CreateVert(floorPointC, &createdVolume);
                MapEdit::Vert *fv3 = MapEdit::CreateVert(floorPointD, &createdVolume);
                MapEdit::Vert *cv0 = MapEdit::CreateVert(ceilPointA, &createdVolume);
                MapEdit::Vert *cv1 = MapEdit::CreateVert(ceilPointB, &createdVolume);
                MapEdit::Vert *cv2 = MapEdit::CreateVert(ceilPointC, &createdVolume);
                MapEdit::Vert *cv3 = MapEdit::CreateVert(ceilPointD, &createdVolume);
                if (heightcomponent < 0)
                {
                    auto temp0 = fv0;
                    auto temp1 = fv1;
                    auto temp2 = fv2;
                    auto temp3 = fv3;
                    fv0 = cv0;
                    fv1 = cv1;
                    fv2 = cv2;
                    fv3 = cv3;
                    cv0 = temp0;
                    cv1 = temp1;
                    cv2 = temp2;
                    cv3 = temp3;
                }

                MapEdit::Edge *f0_to_f1 = MapEdit::CreateEdge(fv0, fv1, &createdVolume);
                MapEdit::Edge *f1_to_f2 = MapEdit::CreateEdge(fv1, fv2, &createdVolume);
                MapEdit::Edge *f2_to_f3 = MapEdit::CreateEdge(fv2, fv3, &createdVolume);
                MapEdit::Edge *f3_to_f0 = MapEdit::CreateEdge(fv3, fv0, &createdVolume);
                MapEdit::Edge *c0_to_c1 = MapEdit::CreateEdge(cv0, cv1, &createdVolume);
                MapEdit::Edge *c1_to_c2 = MapEdit::CreateEdge(cv1, cv2, &createdVolume);
                MapEdit::Edge *c2_to_c3 = MapEdit::CreateEdge(cv2, cv3, &createdVolume);
                MapEdit::Edge *c3_to_c0 = MapEdit::CreateEdge(cv3, cv0, &createdVolume);
                MapEdit::Edge *f0_to_c0 = MapEdit::CreateEdge(fv0, cv0, &createdVolume);
                MapEdit::Edge *f1_to_c1 = MapEdit::CreateEdge(fv1, cv1, &createdVolume);
                MapEdit::Edge *f2_to_c2 = MapEdit::CreateEdge(fv2, cv2, &createdVolume);
                MapEdit::Edge *f3_to_c3 = MapEdit::CreateEdge(fv3, cv3, &createdVolume);

                MapEdit::CreateFace({f0_to_f1, f1_to_f2, f2_to_f3, f3_to_f0}, &createdVolume);
                MapEdit::Face *revthis = MapEdit::CreateFace({c0_to_c1, c1_to_c2, c2_to_c3, c3_to_c0},
                                                             &createdVolume);
                MapEdit::FaceLoopReverse(revthis);
                MapEdit::CreateFace({f0_to_f1, f0_to_c0, f1_to_c1, c0_to_c1}, &createdVolume);
                MapEdit::CreateFace({f1_to_f2, f1_to_c1, c1_to_c2, f2_to_c2}, &createdVolume);
                MapEdit::CreateFace({f2_to_f3, f2_to_c2, c2_to_c3, f3_to_c3}, &createdVolume);
                MapEdit::CreateFace({f3_to_f0, f3_to_c3, c3_to_c0, f0_to_c0}, &createdVolume);

                LevelEditorVolumes.put(createdVolume);

                for (size_t i = 0; i < createdVolume.faces.lenu(); ++i)
                {
                    MapEdit::Face *face = createdVolume.faces[i];
                    MY_VERTEX_BUFFER.clear();
                    TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                    RebindGPUMesh(&face->facemesh, sizeof(float)*MY_VERTEX_BUFFER.size(), MY_VERTEX_BUFFER.data());
                    face->texture = SelectedTexture; 
                }

                simpleBrushToolState = SimpleBrushToolState::NotActive;
            }
            if (KeysPressed[SDL_SCANCODE_ESCAPE])
            {
                simpleBrushToolState = SimpleBrushToolState::NotActive;
            }
            break;
    }
}

void level_editor_t::Draw()
{
    glBindFramebuffer(GL_FRAMEBUFFER, RenderTargetGame.fbo);
    glViewport(0, 0, RenderTargetGame.width, RenderTargetGame.height);
    glClearColor(0.15f, 0.15f, 0.15f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_DEPTH_TEST);

    float aspectratio = float(BackbufferWidth) / float(BackbufferHeight);
    float fovy = HorizontalFOVToVerticalFOV_RadianToRadian(90.f*GM_DEG2RAD, aspectratio);
    ActivePerspectiveMatrix = ProjectionMatrixPerspective(fovy, aspectratio, GAMEPROJECTION_NEARCLIP, GAMEPROJECTION_FARCLIP);
    mat4 perspectiveMatrix = ActivePerspectiveMatrix;
    mat4 viewMatrix = ActiveViewMatrix;

    UseShader(EditorShader_Scene);
    glEnable(GL_CULL_FACE);

    GLBindMatrix4fv(EditorShader_Scene, "projMatrix", 1, perspectiveMatrix.ptr());
    GLBindMatrix4fv(EditorShader_Scene, "viewMatrix", 1, viewMatrix.ptr());

    mat4 modelMatrix = mat4();

    GLBindMatrix4fv(EditorShader_Scene, "modelMatrix", 1, modelMatrix.ptr());

    MapEdit::Face *selectedFace = NULL;
    MapEdit::Face *hoveredFace = NULL;
    for (int i = 0; i < MapEdit::LevelEditorFaces.count; ++i)
    {
        MapEdit::Face *editorVolumeFace = MapEdit::LevelEditorFaces.At(i);
        if (ActiveState == FACE_MANIP)
        {
            if (LevelEditor.SelectedFace == editorVolumeFace)
            {
                selectedFace = LevelEditor.SelectedFace;
                continue;
            }
            else if (editorVolumeFace->hovered)
            {
                // DONT CARE ABOUT HOVERED hoveredFaceMesh = editorVolumeFace->facemesh;
                editorVolumeFace->hovered = false;
                //continue;
            }
        }

        const GPUTexture ftex = editorVolumeFace->texture.gputex;
        const GPUMesh fm = editorVolumeFace->facemesh;
        RenderGPUMesh(fm.idVAO, fm.idVBO, fm.vertexCount, &ftex);
    }

    if (ActiveState == FACE_MANIP)
    {
        if (selectedFace)
        {
            UseShader(EditorShader_FaceSelected);
            float sf = (sinf(TimeSinceStart * 2.7f) + 1.f) / 2.f;
            sf *= 0.1f;
            GLBind3f(EditorShader_FaceSelected, "tint", 1.0f, 1.0f - sf, 1.0f - sf);
            GLBindMatrix4fv(EditorShader_FaceSelected, "projMatrix", 1, perspectiveMatrix.ptr());
            GLBindMatrix4fv(EditorShader_FaceSelected, "viewMatrix", 1, viewMatrix.ptr());
            modelMatrix = mat4();
            GLBindMatrix4fv(EditorShader_FaceSelected, "modelMatrix", 1, modelMatrix.ptr());
            const GPUTexture ftex = selectedFace->texture.gputex;
            const GPUMesh fm = selectedFace->facemesh;
            RenderGPUMesh(fm.idVAO, fm.idVBO, fm.vertexCount, &ftex);
        }
        // else if (hoveredFace)
        // {
        //     UseShader(EditorShader_FaceSelected);
        //     GLBind3f(EditorShader_FaceSelected, "tint", 0.9f, 0.9f, 0.0f);
        //     GLBindMatrix4fv(EditorShader_FaceSelected, "projMatrix", 1, perspectiveMatrix.ptr());
        //     GLBindMatrix4fv(EditorShader_FaceSelected, "viewMatrix", 1, viewMatrix.ptr());
        //     modelMatrix = mat4();
        //     GLBindMatrix4fv(EditorShader_FaceSelected, "modelMatrix", 1, modelMatrix.ptr());
        //     const GPUTexture ftex = hoveredFace->texture.gputex;
        //     const GPUMesh fm = hoveredFace->facemesh;
        //     RenderGPUMesh(fm.idVAO, fm.idVBO, fm.vertexCount, &ftex);
        // }
    }

    // PRIMITIVES

    // Draw outline of selected faces
    for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
    {
        const MapEdit::Volume& volume = LevelEditor.LevelEditorVolumes[SELECTED_MAP_VOLUMES_INDICES.At(i)];
        for (size_t j = 0; j < volume.faces.lenu(); ++j)
        {
            MapEdit::Face *selVolFace = volume.faces[j];
            std::vector<MapEdit::Edge*> faceEdges = selVolFace->GetEdges();
            for (MapEdit::Edge* e : faceEdges)
            {
                SupportRenderer.DrawLine(e->a->pos, e->b->pos, vec4(1,0,0,0.5f), 2.f);
            }
        }
    }
    // Draw vertex handles
    if (ActiveState == VERTEX_MANIP)
    {
        for (int i = 0; i < SELECTABLE_VERTICES.size(); ++i)
        {
            MapEdit::Vert *v = SELECTABLE_VERTICES[i];
            vec4 discHandleColor = vec4(RGBHEXTO1(0xFF8000), 1.f);
            if (std::find(SELECTED_VERTICES.begin(), SELECTED_VERTICES.end(), v) != SELECTED_VERTICES.end())
                discHandleColor = vec4(RGB255TO1(254,8,8),1.f);
            SupportRenderer.DrawSolidDisc(v->pos, LevelEditor.EditorCam.Position - v->pos, GetEditorHandleSize(v->pos, DISC_HANDLE_RADIUS),
                                   discHandleColor);
        }
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    vec3 gridTranslation = vec3(SnapToGrid(LevelEditor.EditorCam.Position.x), 0.f, SnapToGrid(LevelEditor.EditorCam.Position.z));
    mat3 gridRotation = mat3();
    if (GRID_ORIGIN != vec3())
    {
        gridTranslation = GRID_ORIGIN;
        gridRotation = GetGridRotationMatrix();
    }
    SupportRenderer.DrawGrid(GRID_INCREMENT, gridRotation, gridTranslation, &perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));
    SupportRenderer.FlushPrimitives(&perspectiveMatrix, &viewMatrix, RenderTargetGame.depthTexId, vec2((float)RenderTargetGame.width, (float)RenderTargetGame.height));

    // maybe I want billboards to still show up (but faded) when occluded 
    // glEnable(GL_DEPTH_TEST);
    // Entity billboards
    DrawEntityBillboards();
    SupportRenderer.DrawPickableBillboards_GL(ActivePerspectiveMatrix.ptr(), ActiveViewMatrix.ptr(), false);
    SupportRenderer.ClearPickableBillboards();

    // // UseShader(editorShader_Wireframe);
    // // glEnable(GL_CULL_FACE);
    // // GLBindMatrix4fv(editorShader_Wireframe, "projMatrix", 1, perspectiveMatrix.ptr());
    // // GLBindMatrix4fv(editorShader_Wireframe, "viewMatrix", 1, viewMatrix.ptr());
    // // GLBindMatrix4fv(editorShader_Wireframe, "modelMatrix", 1, modelMatrix.ptr());
    // // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    // glEnable(GL_DEPTH_TEST);
    // for (int i = 0; i < MapEdit::EDITOR_FACES.count; ++i)
    // {
    //     MapEdit::Face *editorVolumeFace = MapEdit::EDITOR_FACES.At(i);
    //     std::vector<MapEdit::Edge*> faceEdges = editorVolumeFace->GetEdges();
    //     for (MapEdit::Edge* e : faceEdges)
    //     {
    //         SupportRenderer.DrawLine(e->a->pos, e->b->pos, vec4(1,1,1,1), 1.2f);
    //     }
    //     // const FaceBatch fb = editorVolumeFace->facemesh;
    //     // RenderFaceBatch(fb);
    // }
    // PrimitiveDrawAll(&perspectiveMatrix, &viewMatrix, renderTargetGame.depthTexId, vec2((float)renderTargetGame.width,(float)renderTargetGame.height));
    // if (primitivesDepthTest)
    //     glEnable(GL_DEPTH_TEST);
    // else
    //     glDisable(GL_DEPTH_TEST);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}



#define MAPSER_VOLUMES_U64_DIVIDER_START 0x49779b0ef139ca39
#define MAPSER_VOLUMES_U64_DIVIDER_END   0x9fcf207673e66e63 
#define DESER_FACE_ADDITIONAL_SIZE 60

// TODO(Kevin): add binary format file data
// [4 bytes] : file type identifier
// [4 bytes] : 0x00000000
// [4 bytes] : file serial version 
// [4 bytes] : payload size n
// [n bytes] : payload
// [4 bytes] : EOF identifier

bool level_editor_t::SaveMap(const char *mapFilePath)
{
#if INTERNAL_BUILD
    std::unordered_set<u32> SerPtrSet;
#endif

    ByteBuffer mapbuf;
    ByteBufferInit(&mapbuf);

    ByteBufferWrite(&mapbuf, u64, MAPSER_VOLUMES_U64_DIVIDER_START);
    const int volumesCount = (int)LevelEditorVolumes.lenu();
    ByteBufferWrite(&mapbuf, int, volumesCount);

    for (int i = 0; i < volumesCount; ++i)
    {
        MapEdit::Volume& volume = LevelEditorVolumes[i];
        
        ByteBufferWrite(&mapbuf, u64, volume.persistId);

        ByteBufferWrite(&mapbuf, size_t, volume.verts.lenu());
        for (size_t i = 0; i < volume.verts.lenu(); ++i)
        {
            MapEdit::Vert *v = volume.verts[i];
#if INTERNAL_BUILD
            SerPtrSet.emplace(v->elemId);
#endif
            ByteBufferWrite(&mapbuf, u32, v->elemId);
            ByteBufferWrite(&mapbuf, float, v->pos.x);
            ByteBufferWrite(&mapbuf, float, v->pos.y);
            ByteBufferWrite(&mapbuf, float, v->pos.z);
        }

        ByteBufferWrite(&mapbuf, size_t, volume.edges.lenu());
        for (size_t i = 0; i < volume.edges.lenu(); ++i)
        {
            MapEdit::Edge *e = volume.edges[i];

            u32 ep = e->elemId;
            u32 ea = e->a->elemId;
            u32 eb = e->b->elemId;
#if INTERNAL_BUILD
            ASSERT(SerPtrSet.find(ea) != SerPtrSet.end());
            ASSERT(SerPtrSet.find(eb) != SerPtrSet.end());
            SerPtrSet.emplace(ep);
#endif
            ByteBufferWrite(&mapbuf, u32, ep);
            ByteBufferWrite(&mapbuf, u32, ea);
            ByteBufferWrite(&mapbuf, u32, eb);
        }

        ByteBufferWrite(&mapbuf, size_t, volume.faces.lenu());
        for (size_t i = 0; i < volume.faces.lenu(); ++i)
        {
            MapEdit::Face *f = volume.faces[i];

            // save the sequence of edges in the face's loop cycle by their pointers
            const std::vector<MapEdit::Edge*>& faceEdges = f->GetEdges();
            ByteBufferWrite(&mapbuf, size_t, faceEdges.size());
            for (MapEdit::Edge *fe : faceEdges)
            {
#if INTERNAL_BUILD
                ASSERT(SerPtrSet.find(fe->elemId) != SerPtrSet.end());
#endif
                ByteBufferWrite(&mapbuf, u32, fe->elemId);
            }
            u32 faceTexturePersistId = f->texture.persistId;
            ByteBufferWrite(&mapbuf, u32, faceTexturePersistId);
            // leaving some buffer room for additional Face data in the future
            // make sure to decrement as needed and mirror same value on deserialize
            u8 nullbytes[DESER_FACE_ADDITIONAL_SIZE];
            ByteBufferWriteBulk(&mapbuf, &nullbytes, DESER_FACE_ADDITIONAL_SIZE);
        }
    }

    ByteBufferWrite(&mapbuf, u64, MAPSER_VOLUMES_U64_DIVIDER_END);

    ByteBufferWrite(&mapbuf, size_t, LevelEntities.lenu());
    for (size_t i = 0; i < LevelEntities.lenu(); ++i)
    {
        level_entity_t& Ent = LevelEntities[i];
        Ent.SerializeToEditableMapFile(&mapbuf);
    }

    bool writtenToFile = ByteBufferWriteToFile(&mapbuf, mapFilePath) == 1;

    ByteBufferFree(&mapbuf);

    return writtenToFile;
}

bool level_editor_t::LoadMap(const char *mapFilePath)
{
    std::unordered_map<u32, void*> DeserElemIdToElem;

    ByteBuffer mapbuf;
    if (ByteBufferReadFromFile(&mapbuf, mapFilePath) == 0)
        return false;

    u64 u64slot;
    ByteBufferRead(&mapbuf, u64, &u64slot);
    ASSERT(u64slot == MAPSER_VOLUMES_U64_DIVIDER_START);

    int volumesCount = -1;
    ByteBufferRead(&mapbuf, int, &volumesCount);

    for (int i = 0; i < volumesCount; ++i)
    {
        MapEdit::Volume owner;
        ByteBufferRead(&mapbuf, u64, &owner.persistId);
        MapEdit::session_VolumePersistIdCounter = GM_max(owner.persistId, MapEdit::session_VolumePersistIdCounter);

        size_t vertexCount, edgeCount, faceCount = 0;

        ByteBufferRead(&mapbuf, size_t, &vertexCount);
        for (size_t j = 0; j < vertexCount; ++j)
        {
            u32 vElemId;
            vec3 vpos;
            ByteBufferRead(&mapbuf, u32, &vElemId);
            ByteBufferRead(&mapbuf, float, &vpos.x);
            ByteBufferRead(&mapbuf, float, &vpos.y);
            ByteBufferRead(&mapbuf, float, &vpos.z);

            void *elemptr = (void*)CreateVert(vpos, &owner); // MEM ALLOC
            DeserElemIdToElem.emplace(vElemId, elemptr);
        }

        ByteBufferRead(&mapbuf, size_t, &edgeCount);
        for (size_t j = 0; j < edgeCount; ++j)
        {
            u32 eElemId;
            u32 aElemId;
            u32 bElemId;
            ByteBufferRead(&mapbuf, u32, &eElemId);
            ByteBufferRead(&mapbuf, u32, &aElemId);
            ByteBufferRead(&mapbuf, u32, &bElemId);

            MapEdit::Vert *av = (MapEdit::Vert*)DeserElemIdToElem.at(aElemId);
            MapEdit::Vert *bv = (MapEdit::Vert*)DeserElemIdToElem.at(bElemId);
            void *elemptr = (void*)CreateEdge(av, bv, &owner); // MEM ALLOC
            DeserElemIdToElem.emplace(eElemId, elemptr);
        }

        ByteBufferRead(&mapbuf, size_t, &faceCount);
        for (size_t j = 0; j < faceCount; ++j)
        {
            size_t faceEdgesCount = 0;
            std::vector<MapEdit::Edge*> faceEdges;
            
            ByteBufferRead(&mapbuf, size_t, &faceEdgesCount);
            for (int k = 0; k < faceEdgesCount; ++k)
            {
                u32 feElemId;
                ByteBufferRead(&mapbuf, u32, &feElemId);
                faceEdges.push_back((MapEdit::Edge*)DeserElemIdToElem.at(feElemId));
            }

            u32 faceTexturePersistId;
            ByteBufferRead(&mapbuf, u32, &faceTexturePersistId);
            ByteBufferAdvancePosition(&mapbuf, DESER_FACE_ADDITIONAL_SIZE);

            MapEdit::Face *face = CreateFace(faceEdges, &owner); // MEM ALLOC

            MY_VERTEX_BUFFER.clear();
            TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER); // TODO(Kevin): do smarter triangulation...
            RebindGPUMesh(&face->facemesh, sizeof(float)*MY_VERTEX_BUFFER.size(), MY_VERTEX_BUFFER.data());

            face->texture = Assets.GetTextureById(faceTexturePersistId);
        }

        LevelEditorVolumes.put(owner);
    }

    ByteBufferRead(&mapbuf, u64, &u64slot);
    ASSERT(u64slot == MAPSER_VOLUMES_U64_DIVIDER_END);

     size_t LevelEntitiesCount = 0;
     ByteBufferRead(&mapbuf, size_t, &LevelEntitiesCount);
     for (size_t i = 0; i < LevelEntitiesCount; ++i)
     {
         level_entity_t Ent;
         Ent.DeserializeFromEditableMapFile(&mapbuf);
         LevelEntities.put(Ent);
     }

    ByteBufferFree(&mapbuf);

    return true;
}




