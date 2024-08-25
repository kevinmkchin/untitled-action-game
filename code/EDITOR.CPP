#pragma region MAP_EDITOR_CODE

std::vector<float> MY_VERTEX_BUFFER;

vec3 editorCameraPosition = vec3(128, 128, 128);
float GRID_INCREMENT = 8.f;
float DISC_HANDLE_RADIUS = 10.f;
NiceArray<MapEdit::Volume, 3200> LOADED_MAP_VOLUMES;
NiceArray<int, 16> SELECTED_MAP_VOLUMES_INDICES;
std::vector<MapEdit::Vert*> SELECTABLE_VERTICES;
std::vector<MapEdit::Vert*> SELECTED_VERTICES;
std::vector<int> SELECTED_SELECTABLE_VERTICES_INDICES;
std::vector<MapEdit::Face*> SELECTABLE_FACES;


enum class MapEditorTools
{
    SimpleBrushTool,
    VertexManip,
    EdgeManip,
    FaceManip,
};
MapEditorTools editorActiveTool = MapEditorTools::SimpleBrushTool;

enum class SimpleBrushToolState
{
    NotAvailable,
    NotActive,
    DrawingRectangle,
    DrawingHeight
};
SimpleBrushToolState simpleBrushToolState = SimpleBrushToolState::NotActive;


vec3 ScreenPointToWorldRay(ivec2 screenspaceCoords)
{
    // https://antongerdelan.net/opengl/raycasting.html
    // assuming 3D world view is taking up the entire window
    // Reversing perspective divide not necessary because this is a vector/direction/ray with no intrinsic depth.
    float x_NDC = ((float)screenspaceCoords.x / (float)BACKBUFFER_WIDTH) * 2.f - 1.f;
    float y_NDC = (float(BACKBUFFER_HEIGHT - screenspaceCoords.y) / (float)BACKBUFFER_HEIGHT) * 2.f - 1.f;
    vec4 ray_in_clipspace = vec4(x_NDC, y_NDC, -1.f, 1.f);
    vec4 ray_in_viewspace = activePerspectiveMatrix.GetInverse() * ray_in_clipspace;
    ray_in_viewspace = vec4(ray_in_viewspace.x, ray_in_viewspace.y, -1.f, 0.f);
    vec3 ray_in_worldspace = Normalize((activeViewMatrix.GetInverse() * ray_in_viewspace).xyz);
    return ray_in_worldspace;
}

vec3 ScreenPointToWorldPoint(ivec2 screenspaceCoords, float z_NDC)
{
    float x_NDC = ((float)screenspaceCoords.x / (float)BACKBUFFER_WIDTH) * 2.f - 1.f;
    float y_NDC = (float(BACKBUFFER_HEIGHT - screenspaceCoords.y) / (float)BACKBUFFER_HEIGHT) * 2.f - 1.f;
    vec4 point_in_clipspace = vec4(x_NDC, y_NDC, z_NDC, 1.f);
    // For points, reverse perspective divide after the inverse projection matrix transformation because it's easier that way.
    vec4 point_in_viewspace_before_perspective_divide = activePerspectiveMatrix.GetInverse() * point_in_clipspace;
    vec4 point_in_viewspace = point_in_viewspace_before_perspective_divide / point_in_viewspace_before_perspective_divide.w;
    vec4 point_in_worldspace = activeViewMatrix.GetInverse() * point_in_viewspace;
    return point_in_worldspace.xyz;
}

vec3 WorldPointToScreenPoint(vec3 worldPosition)
{
    vec4 clipspaceCoordinates = activePerspectiveMatrix * activeViewMatrix * vec4(worldPosition, 1.f);
    float screenspaceRatioX = ((clipspaceCoordinates.x / clipspaceCoordinates.w) + 1.f) / 2.f;
    float screenspaceRatioY = 1.f - (((clipspaceCoordinates.y / clipspaceCoordinates.w) + 1.f) / 2.f);
    float internalResolutionWidth = (float)BACKBUFFER_WIDTH;
    float internalResolutionHeight = (float)BACKBUFFER_HEIGHT;
    float distanceFromCameraWCS = Dot(worldPosition - editorCameraPosition, cameraDirection);
    return vec3(screenspaceRatioX * internalResolutionWidth, screenspaceRatioY * internalResolutionHeight, distanceFromCameraWCS);
}

float SnapToMapEditorGrid(float beforeSnap)
{
    int adjustedBeforeSnap = int(roundf(beforeSnap) + ((beforeSnap < 0.f ? -0.5f : 0.5f) * GRID_INCREMENT));
    // TODO(Kevin): fix divide by zero when GRID_INCREMENT is less than 1
    int snapped = adjustedBeforeSnap - adjustedBeforeSnap % (int)(GRID_INCREMENT);
    return (float)snapped;
}

vec3 SnapToMapEditorGrid(vec3 beforeSnap)
{
    return vec3(SnapToMapEditorGrid(beforeSnap.x), SnapToMapEditorGrid(beforeSnap.y), SnapToMapEditorGrid(beforeSnap.z));
}

float GetEditorHandleSize(vec3 worldPosition, float sizeInPixels)
{
    float distanceFromCamera = Dot(worldPosition - editorCameraPosition, cameraDirection);
    quat camOrientation = EulerToQuat(cameraRotation * GM_DEG2RAD);
    vec3 screenPos = WorldPointToScreenPoint(editorCameraPosition + RotateVector(vec3(distanceFromCamera, 0, 0), camOrientation));
    vec3 screenPos2 = WorldPointToScreenPoint(editorCameraPosition + RotateVector(vec3(distanceFromCamera, 0, 32.f), camOrientation));
    // scaled by 32 to avoid floating point imprecision
    float screenDist = Magnitude(screenPos - screenPos2);
    return (sizeInPixels*32.f / GM_max(screenDist, 0.0001f));
}

void TickMapEditor()
{
    SDL_SetRelativeMouseMode(mousecurrent & SDL_BUTTON(SDL_BUTTON_RIGHT) ? SDL_TRUE : SDL_FALSE);

    // EDITOR CAMERA MOVE
    if (mousecurrent & SDL_BUTTON(SDL_BUTTON_RIGHT))
    {
        float camYawDelta = mousedelta.x*0.1f;
        float camPitchDelta = mousedelta.y*0.1f;
        cameraRotation.y -= camYawDelta;
        cameraRotation.z -= camPitchDelta;
        if (cameraRotation.z > 89.f)
            cameraRotation.z = 89.f;
        if (cameraRotation.z < -89.f)
            cameraRotation.z = -89.f;
    }
    cameraDirection = Normalize(OrientationToDirection(EulerToQuat(cameraRotation * GM_DEG2RAD)));
    cameraRight = Normalize(Cross(cameraDirection, GM_UP_VECTOR));
    cameraUp = Normalize(Cross(cameraRight, cameraDirection));
    float moveSpeed = 250.f;
    if (keyscurrent[SDL_SCANCODE_LSHIFT])
        moveSpeed *= 2.7f;
    vec3 playerPositionDelta;
    if (mousecurrent & SDL_BUTTON(SDL_BUTTON_RIGHT))
    {
        if (keyscurrent[SDL_SCANCODE_W])
            playerPositionDelta += cameraDirection * moveSpeed * g_DeltaTime;
        if (keyscurrent[SDL_SCANCODE_A])
            playerPositionDelta += -cameraRight * moveSpeed * g_DeltaTime;
        if (keyscurrent[SDL_SCANCODE_S])
            playerPositionDelta += -cameraDirection * moveSpeed * g_DeltaTime;
        if (keyscurrent[SDL_SCANCODE_D])
            playerPositionDelta += cameraRight * moveSpeed * g_DeltaTime;
        if (keyscurrent[SDL_SCANCODE_Q])
            playerPositionDelta += -GM_UP_VECTOR * moveSpeed * g_DeltaTime;
        if (keyscurrent[SDL_SCANCODE_E])
            playerPositionDelta += GM_UP_VECTOR * moveSpeed * g_DeltaTime;
        editorCameraPosition += playerPositionDelta;
    }
    activeViewMatrix = ViewMatrixLookAt(editorCameraPosition, editorCameraPosition + cameraDirection, cameraUp);

    // DRAW AXIS AND GRID LINES
    vec3 intersection_with_xz_plane;
    if (IntersectPlaneAndLine(vec3(), GM_UP_VECTOR, editorCameraPosition,
                              -GM_UP_VECTOR,
                              &intersection_with_xz_plane))
    {
        float gridVisualIncrement = GRID_INCREMENT;
        intersection_with_xz_plane = SnapToMapEditorGrid(intersection_with_xz_plane);
        float gridoriginx = intersection_with_xz_plane.x;
        float gridoriginz = intersection_with_xz_plane.z;
        int linedrawcount = 4000 / (int)gridVisualIncrement;
        for (int i = -linedrawcount; i < linedrawcount; ++i)
        {
            if (float(i) * gridVisualIncrement + gridoriginz != 0.f)
                PrimitiveDrawLine(vec3(320000.f + gridoriginx, 0.f, float(i) * gridVisualIncrement + gridoriginz),
                                  vec3(-320000.f + gridoriginx, 0.f, float(i) * gridVisualIncrement + gridoriginz),
                                  vec4(RGB255TO1(155,155,155), 0.25f));
            if (float(i) * gridVisualIncrement + gridoriginx != 0.f)
                PrimitiveDrawLine(vec3(float(i) * gridVisualIncrement + gridoriginx, 0.f, 320000.f + gridoriginz),
                                  vec3(float(i) * gridVisualIncrement + gridoriginx, 0.f, -320000.f + gridoriginz),
                                  vec4(RGB255TO1(155,155,155), 0.25f));
        }
    }
    PrimitiveDrawLine(vec3(320000.f, 0.f, 0.f), vec3(-320000.f, 0.f, 0.f), vec4(RGB255TO1(205, 56, 9), 0.8f));
    // PrimitiveDrawLine(vec3(0.f, 320000.f, 0.f), vec3(0.f, -320000.f, 0.f), vec4(RGB255TO1(67, 123, 9), 0.8f));
    PrimitiveDrawLine(vec3(0.f, 0.f, 320000.f), vec3(0.f, 0.f, -320000.f), vec4(RGB255TO1(21, 129, 205), 0.8f));
    Gui::PrimitiveText(100,100, 36, Gui::Align::Left, std::to_string(GRID_INCREMENT).c_str());

    if (Gui::PrimitiveLabelledButton(Gui::UIRect(400,10,60,32), "Grid -", Gui::Align::Right))
    {
        GRID_INCREMENT /= 2.f;
    }
    if (Gui::PrimitiveLabelledButton(Gui::UIRect(470,10,60,32), "Grid +", Gui::Align::Left))
    {
        GRID_INCREMENT *= 2.f;
    }


    static u32 hotHandleId = 0;


    if (keyspressed[SDL_SCANCODE_1] ||
        Gui::ImageButton(Gui::UIRect(10,10,32,32), tex_Default.id, tex_Default.id, tex_Default.id))
    {
        editorActiveTool = MapEditorTools::SimpleBrushTool;
        hotHandleId = 0;
    }
    if (keyspressed[SDL_SCANCODE_2] ||
        Gui::ImageButton(Gui::UIRect(10,52,32,32), tex_Default.id, tex_Default.id, tex_Default.id))
    {
        editorActiveTool = MapEditorTools::VertexManip;
        hotHandleId = 0;
    }
    if (keyspressed[SDL_SCANCODE_3] ||
        Gui::ImageButton(Gui::UIRect(10,94,32,32), tex_Default.id, tex_Default.id, tex_Default.id))
    {
        editorActiveTool = MapEditorTools::EdgeManip;
        hotHandleId = 0;
    }
    if (keyspressed[SDL_SCANCODE_4] ||
        Gui::ImageButton(Gui::UIRect(10,136,32,32), tex_Default.id, tex_Default.id, tex_Default.id))
    {
        editorActiveTool = MapEditorTools::FaceManip;
        hotHandleId = 0;
    }

    Gui::PrimitiveText(100,28, 9, Gui::Align::Left, "Toggle Auto Colliders []");

    if (Gui::anyElementActive)
        return;

    // Selection handles

    // on mouse up, disc remains selected
    // on mouse down, if disc not selected, then if ctrl pressed then add to selection but don't start drag
    //                                      if ctrl not pressed then start drag
    // on mouse down, if disc was already selected, then if ctrl pressed then deselect that disc/vert, if ctrl not pressed
    // then start checking for drag

    SELECTABLE_VERTICES.clear();
    SELECTABLE_FACES.clear();
    for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
    {
        MapEdit::Volume volume = LOADED_MAP_VOLUMES.At(SELECTED_MAP_VOLUMES_INDICES.At(i));

        for (int j = 0; j < volume.verts.size(); ++j)
        {
            SELECTABLE_VERTICES.push_back(volume.verts[j]);
        }

        for (int j = 0; j < volume.faces.size(); ++j)
        {
            SELECTABLE_FACES.push_back(volume.faces[j]);
        }
    }

    if (editorActiveTool == MapEditorTools::FaceManip)
    {
        static vec3 intersectOffsetFromLoopbaseVert;

        bool lmbPressedThisFrame = mousepressed & SDL_BUTTON(SDL_BUTTON_LEFT);
        bool lmbIsPressed = mousecurrent & SDL_BUTTON(SDL_BUTTON_LEFT);

        for (u32 id = 1; id <= SELECTABLE_FACES.size(); ++id)
        {
            MapEdit::Face *face = SELECTABLE_FACES[id-1];
            vec3 idrgb = HandleIdToRGB(id);
            int count;
            MapEdit::TriangulateFace_QuickDumb_WithColor(*face, idrgb, HANDLES_VB.data + HANDLES_VB.count, &count);
            HANDLES_VB.count += count;
        }
        u32 hoveredFaceId = FlushHandles(mousepos, renderTargetGame, activeViewMatrix, activePerspectiveMatrix, false);
        LogMessage("%d", hoveredFaceId);

        if (hoveredFaceId > 0)
        {
            MapEdit::Face *face = SELECTABLE_FACES[hoveredFaceId-1];
            face->hovered = true;

            if (lmbPressedThisFrame)
            {
                hotHandleId = hoveredFaceId;
                SELECTED_VERTICES = face->GetVertices();

                vec3 worldpos_mouse = ScreenPointToWorldPoint(mousepos, 0.f);
                vec3 worldray_mouse = ScreenPointToWorldRay(mousepos);
                vec3 intersect;
                IntersectPlaneAndLineWithDirections(face->loopbase->v->pos, face->QuickNormal(), worldpos_mouse, worldray_mouse, &intersect);
                intersectOffsetFromLoopbaseVert = intersect - face->loopbase->v->pos;
            }
        }


        if (lmbIsPressed && hotHandleId > 0)
        {
            MapEdit::Face *hotFace = SELECTABLE_FACES[hotHandleId-1];

            vec3 worldpos_mouse = ScreenPointToWorldPoint(mousepos, 0.f);
            vec3 worldray_mouse = ScreenPointToWorldRay(mousepos);

            vec3 translation;
            vec3 faceNormal = hotFace->QuickNormal();
            // PrimitiveDrawSolidDisc(intersect, -cameraDirection, 8.f, vec4(1.0,0.0,1.0,1.0));
            vec3 dragPlanePoint = hotFace->loopbase->v->pos + intersectOffsetFromLoopbaseVert;
            vec3 intersect;
            IntersectPlaneAndLineWithDirections(dragPlanePoint, -cameraDirection, worldpos_mouse, worldray_mouse, &intersect);
            float scaleComponent = Dot((intersect - dragPlanePoint), faceNormal);
            translation = SnapToMapEditorGrid(scaleComponent) * faceNormal;//vec3(0.f,yTranslation,0.f);

            // vec3 dragPlanePoint = hotFace->pos;
            // vec3 dragPlaneNormal = -cameraDirection;
            // vec3 intersect;
            // IntersectPlaneAndLineWithDirections(dragPlanePoint, dragPlaneNormal, worldpos_mouse, worldray_mouse, &intersect);
            // intersect = vec3(intersect.x, SnapToMapEditorGrid(intersect.y), intersect.z);
            // float yTranslation = Dot((intersect - dragPlanePoint), GM_UP_VECTOR);
            // translation = vec3(0.f,yTranslation,0.f);
            // if (keyscurrent[SDL_SCANCODE_LALT])
            // {
            //     vec3 dragPlanePoint = hotVert->pos;
            //     vec3 dragPlaneNormal = -cameraDirection;
            //     vec3 intersect;
            //     IntersectPlaneAndLineWithDirections(dragPlanePoint, dragPlaneNormal, worldpos_mouse, worldray_mouse, &intersect);
            //     intersect = vec3(intersect.x, SnapToMapEditorGrid(intersect.y), intersect.z);
            //     float yTranslation = Dot((intersect - dragPlanePoint), GM_UP_VECTOR);
            //     translation = vec3(0.f,yTranslation,0.f);
            // }
            // else
            // {
            //     vec3 XZDragPlanePoint = hotVert->pos;
            //     vec3 XZDragPlaneNormal = GM_UP_VECTOR;
            //     vec3 intersect;
            //     IntersectPlaneAndLine(XZDragPlanePoint, XZDragPlaneNormal, worldpos_mouse, worldray_mouse, &intersect);
            //     translation = SnapToMapEditorGrid(intersect) - hotVert->pos;
            // }
            // if (translation != vec3())
            // {
            //     TranslationHappenedForSomeFrames = true;
            //     TotalTranslationThisMouseDown += translation;
            // }

            for (MapEdit::Vert *vert : SELECTED_VERTICES)
            {
                vert->pos += translation;
            }
            for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
            {
                MapEdit::Volume& selectedVol = LOADED_MAP_VOLUMES.At(SELECTED_MAP_VOLUMES_INDICES.At(i));
                for (MapEdit::Face *face : selectedVol.faces)
                {
                    MY_VERTEX_BUFFER.clear();
                    MapEdit::TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                    RebindFaceBatchBufferObject(&face->facemesh, (u32)(sizeof(float)*MY_VERTEX_BUFFER.size()), MY_VERTEX_BUFFER.data());
                }
            }
        }
    }

    if (editorActiveTool == MapEditorTools::VertexManip)
    {
        static bool LCtrlDownOnLeftMouseDown = false;
        static bool AlreadySelectedOnLeftMouseDown = false;
        static bool TranslationHappenedForSomeFrames = false;
        static vec3 TotalTranslationThisMouseDown = vec3();

        if (mousepressed & SDL_BUTTON(SDL_BUTTON_LEFT))
        {
            TotalTranslationThisMouseDown = vec3();
            TranslationHappenedForSomeFrames = false;
            LCtrlDownOnLeftMouseDown = keyscurrent[SDL_SCANCODE_LCTRL];

            for (u32 id = 1; id <= SELECTABLE_VERTICES.size(); ++id)
            {
                MapEdit::Vert *vert = SELECTABLE_VERTICES[id-1];
                DoDiscHandle(id, vert->pos, editorCameraPosition, GetEditorHandleSize(vert->pos, DISC_HANDLE_RADIUS + 4.f));
            }
            u32 clickedId = FlushHandles(mousepos, renderTargetGame, activeViewMatrix, activePerspectiveMatrix, false);

            if (clickedId > 0)
            {
                hotHandleId = clickedId;
                auto clickedIter = std::find(SELECTED_SELECTABLE_VERTICES_INDICES.begin(), SELECTED_SELECTABLE_VERTICES_INDICES.end(), clickedId);
                AlreadySelectedOnLeftMouseDown = clickedIter != SELECTED_SELECTABLE_VERTICES_INDICES.end();
                if (!AlreadySelectedOnLeftMouseDown)
                {
                    if (!LCtrlDownOnLeftMouseDown)
                        SELECTED_SELECTABLE_VERTICES_INDICES.clear();
                    SELECTED_SELECTABLE_VERTICES_INDICES.push_back(clickedId);
                }
            }
            else if (!LCtrlDownOnLeftMouseDown)
            {
                SELECTED_SELECTABLE_VERTICES_INDICES.clear();
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

        if (mousecurrent & SDL_BUTTON(SDL_BUTTON_LEFT) && hotHandleId > 0)
        {
            MapEdit::Vert *hotVert = SELECTABLE_VERTICES[hotHandleId-1];

            vec3 worldpos_mouse = ScreenPointToWorldPoint(mousepos, 0.f);
            vec3 worldray_mouse = ScreenPointToWorldRay(mousepos);

            vec3 translation;
            if (keyscurrent[SDL_SCANCODE_LALT])
            {
                vec3 dragPlanePoint = hotVert->pos;
                vec3 dragPlaneNormal = -cameraDirection;
                vec3 intersect;
                IntersectPlaneAndLineWithDirections(dragPlanePoint, dragPlaneNormal, worldpos_mouse, worldray_mouse, &intersect);
                intersect = vec3(intersect.x, SnapToMapEditorGrid(intersect.y), intersect.z);
                float yTranslation = Dot((intersect - dragPlanePoint), GM_UP_VECTOR);
                translation = vec3(0.f,yTranslation,0.f);
            }
            else
            {
                vec3 XZDragPlanePoint = hotVert->pos;
                vec3 XZDragPlaneNormal = GM_UP_VECTOR;
                vec3 intersect;
                IntersectPlaneAndLine(XZDragPlanePoint, XZDragPlaneNormal, worldpos_mouse, worldray_mouse, &intersect);
                translation = SnapToMapEditorGrid(intersect) - hotVert->pos;
            }
            if (translation != vec3())
            {
                TranslationHappenedForSomeFrames = true;
                TotalTranslationThisMouseDown += translation;
            }

            for (int index : SELECTED_SELECTABLE_VERTICES_INDICES)
            {
                MapEdit::Vert *vert = SELECTABLE_VERTICES[index - 1];
                vert->pos += translation;
            }
            for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
            {
                MapEdit::Volume& selectedVol = LOADED_MAP_VOLUMES.At(SELECTED_MAP_VOLUMES_INDICES.At(i));
                for (MapEdit::Face *face : selectedVol.faces)
                {
                    MY_VERTEX_BUFFER.clear();
                    MapEdit::TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                    RebindFaceBatchBufferObject(&face->facemesh, (u32)(sizeof(float)*MY_VERTEX_BUFFER.size()), MY_VERTEX_BUFFER.data());
                }
            }
        }

        if (keyscurrent[SDL_SCANCODE_ESCAPE])
        {
            for (int index : SELECTED_SELECTABLE_VERTICES_INDICES)
            {
                MapEdit::Vert *vert = SELECTABLE_VERTICES[index - 1];
                vert->pos -= TotalTranslationThisMouseDown;
            }
            for (int i = 0; i < SELECTED_MAP_VOLUMES_INDICES.count; ++i)
            {
                MapEdit::Volume& selectedVol = LOADED_MAP_VOLUMES.At(SELECTED_MAP_VOLUMES_INDICES.At(i));
                for (MapEdit::Face *face : selectedVol.faces)
                {
                    MY_VERTEX_BUFFER.clear();
                    MapEdit::TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                    RebindFaceBatchBufferObject(&face->facemesh, (u32)(sizeof(float)*MY_VERTEX_BUFFER.size()), MY_VERTEX_BUFFER.data());
                }
            }

            hotHandleId = 0;
            TotalTranslationThisMouseDown = vec3();
            AlreadySelectedOnLeftMouseDown = false;
            LCtrlDownOnLeftMouseDown = false;
            TranslationHappenedForSomeFrames = false;
        }

        if (mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT))
        {
            if (hotHandleId > 0)
            {
                if (LCtrlDownOnLeftMouseDown)
                {
                    auto clickedIter = std::find(SELECTED_SELECTABLE_VERTICES_INDICES.begin(), SELECTED_SELECTABLE_VERTICES_INDICES.end(), hotHandleId);
                    if (clickedIter != SELECTED_SELECTABLE_VERTICES_INDICES.end())
                    {
                        if (AlreadySelectedOnLeftMouseDown && !TranslationHappenedForSomeFrames)
                            SELECTED_SELECTABLE_VERTICES_INDICES.erase(clickedIter);
                    }
                    else
                    {
                        LogError("wtf");
                    }
                }
                else if (!TranslationHappenedForSomeFrames)
                {
                    SELECTED_SELECTABLE_VERTICES_INDICES.clear();
                    SELECTED_SELECTABLE_VERTICES_INDICES.push_back(hotHandleId);
                }
            }

            hotHandleId = 0;
            TotalTranslationThisMouseDown = vec3();
            AlreadySelectedOnLeftMouseDown = false;
            LCtrlDownOnLeftMouseDown = false;
            TranslationHappenedForSomeFrames = false;
        }
    }

    if (editorActiveTool == MapEditorTools::SimpleBrushTool)
    {
        static vec3 rectstartpoint;
        static vec3 rectendpoint;
        static float desiredBrushY_in_world;
        switch (simpleBrushToolState)
        {
            case SimpleBrushToolState::NotActive:
                if (mousepressed & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    // todo with drawing brush ON other volumes, do picking first to see on which volume we are drawing,
                    // then do the mouse ray intersect...although this optimization might not be needed

                    // We only want to draw within a certain radius from the camera. never too far from the camera.
                    desiredBrushY_in_world = 0.f;
                    vec3 ws = ScreenPointToWorldPoint(mousepos, 0.f);
                    vec3 wr = ScreenPointToWorldRay(mousepos);
                    float f = (desiredBrushY_in_world - ws.y) / wr.y;
                    if (f > 0.f) // should accept any f... since we want to draw brush in any direction
                    {
                        simpleBrushToolState = SimpleBrushToolState::DrawingRectangle;
                        rectstartpoint = SnapToMapEditorGrid(ws + wr * f);
                    }
                }
                break;
            case SimpleBrushToolState::DrawingRectangle:
                if (mousecurrent & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    vec3 ws = ScreenPointToWorldPoint(mousepos, 0.f);
                    vec3 wr = ScreenPointToWorldRay(mousepos);
                    float f = (desiredBrushY_in_world - ws.y) / wr.y;
                    vec3 endpoint = SnapToMapEditorGrid(ws + wr * f);
                    vec3 startToEndVector = vec3(endpoint - rectstartpoint);
                    float d = Magnitude(startToEndVector);
                    vec3 startToEndProjOnToZDirection = Dot(startToEndVector, GM_RIGHT_VECTOR) * GM_RIGHT_VECTOR;
                    vec3 startToEndProjOnToXDirection = Dot(startToEndVector, GM_FORWARD_VECTOR) * GM_FORWARD_VECTOR;
                    PrimitiveDrawLine(rectstartpoint, rectstartpoint + startToEndProjOnToXDirection,
                                      vec4(RGBHEXTO1(0xff8000), 1.0));
                    PrimitiveDrawLine(rectstartpoint, rectstartpoint + startToEndProjOnToZDirection,
                                      vec4(RGBHEXTO1(0xff8000), 1.0));
                    PrimitiveDrawLine(endpoint, endpoint - startToEndProjOnToXDirection,
                                      vec4(RGBHEXTO1(0xff8000), 1.0));
                    PrimitiveDrawLine(endpoint, endpoint - startToEndProjOnToZDirection,
                                      vec4(RGBHEXTO1(0xff8000), 1.0));
                }
                if (mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    vec3 ws = ScreenPointToWorldPoint(mousepos, 0.f);
                    vec3 wr = ScreenPointToWorldRay(mousepos);
                    float f = (desiredBrushY_in_world - ws.y) / wr.y;
                    rectendpoint = SnapToMapEditorGrid(ws + wr * f);
                    vec3 startToEndVector = vec3(rectendpoint - rectstartpoint);
                    float d = Magnitude(startToEndVector);
                    if (true) // TODO check X and Z values to see if theyre bigger than grid size
                        simpleBrushToolState = SimpleBrushToolState::DrawingHeight;
                    else
                        simpleBrushToolState = SimpleBrushToolState::NotActive;
                }
                if (keyspressed[SDL_SCANCODE_ESCAPE])
                    simpleBrushToolState = SimpleBrushToolState::NotActive;

                break;
            case SimpleBrushToolState::DrawingHeight:
                vec3 startToEndVector = vec3(rectendpoint - rectstartpoint);
                float zcomponent = Dot(startToEndVector, GM_RIGHT_VECTOR);
                float xcomponent = Dot(startToEndVector, GM_FORWARD_VECTOR);
                vec3 startToEndProjOnToZDirection = zcomponent * GM_RIGHT_VECTOR;
                vec3 startToEndProjOnToXDirection = xcomponent * GM_FORWARD_VECTOR;

                // let plane be defined at point rectendpoint with normal -cameraDirection
                // then the point of intersection between mouse ray and the plane - endrectpoint and project it onto
                // the direction of translation e.g. GM_UP_VECTOR
                vec3 drawingSurfaceNormal = GM_UP_VECTOR;
                static vec3 height = vec3();
                static vec3 heightBeforeSnap = vec3();
                vec3 pn = -cameraDirection;
                vec3 pp = rectendpoint + heightBeforeSnap;
                vec3 wp = ScreenPointToWorldPoint(mousepos, 0.f);
                vec3 wr = ScreenPointToWorldRay(mousepos);
                vec3 intersection;
                IntersectPlaneAndLineWithDirections(pp, pn, wp, wr, &intersection);
                float trueHeightComp = Dot((intersection - rectendpoint), drawingSurfaceNormal);
                float heightcomponent = SnapToMapEditorGrid(trueHeightComp);
                height = heightcomponent * drawingSurfaceNormal;
                heightBeforeSnap = trueHeightComp * drawingSurfaceNormal;

                vec3 floorPointA = rectstartpoint;
                vec3 floorPointB = rectstartpoint + startToEndProjOnToXDirection;
                vec3 floorPointC = rectendpoint;
                vec3 floorPointD = rectendpoint - startToEndProjOnToXDirection;
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

                PrimitiveDrawLine(floorPointA, floorPointB, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointA, floorPointD, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointC, floorPointB, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointC, floorPointD, vec4(RGBHEXTO1(0xff8000), 1.0));

                PrimitiveDrawLine(floorPointA, ceilPointA, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointB, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointC, ceilPointC, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(floorPointD, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0));

                PrimitiveDrawLine(ceilPointA, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(ceilPointA, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(ceilPointC, ceilPointB, vec4(RGBHEXTO1(0xff8000), 1.0));
                PrimitiveDrawLine(ceilPointC, ceilPointD, vec4(RGBHEXTO1(0xff8000), 1.0));

                if (mousereleased & SDL_BUTTON(SDL_BUTTON_LEFT))
                {
                    // if heigth component is zero or too small then just default to current grid increment

                    MapEdit::Volume createdVolume;

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

                    LOADED_MAP_VOLUMES.PushBack(createdVolume);
                    SELECTED_MAP_VOLUMES_INDICES.PushBack(LOADED_MAP_VOLUMES.count - 1);

                    for (auto face: createdVolume.faces)
                    {
                        MY_VERTEX_BUFFER.clear();
                        TriangulateFace_QuickDumb(*face, &MY_VERTEX_BUFFER);
                        RebindFaceBatchBufferObject(&face->facemesh, (u32) MY_VERTEX_BUFFER.size() * sizeof(float),
                            MY_VERTEX_BUFFER.data());
                        face->facemesh.sharedTexture = tex_Default;
                    }

                    simpleBrushToolState = SimpleBrushToolState::NotActive;
                }
                if (keyspressed[SDL_SCANCODE_ESCAPE])
                {
                    simpleBrushToolState = SimpleBrushToolState::NotActive;
                }
                break;
        }
    }
}


// save editable map

// load editable map

// build/save game map

// load game map


#pragma endregion MAP_EDITOR_CODE
