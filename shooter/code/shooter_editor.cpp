#include "shooter_editor.h"

#include "shooter_editmesh.cpp"

internal void
ScreenPointToWorldPointAndRay(vec2 ScreenPoint, renderer *RenderAPI,
                              vec3 *WorldPoint, vec3 *WorldRay)
{
    u32 BackBufferW, BackBufferH;
    mat4 PerspectiveMatrix, ViewMatrix;
    RenderAPI->GetRenderTargetDimensions(&BackBufferW, &BackBufferH);
    RenderAPI->GetViewAndProjMatrices(&PerspectiveMatrix, &ViewMatrix);
    mat4 InvPerspectiveMatrix = PerspectiveMatrix.GetInverse();
    mat4 InvViewMatrix = ViewMatrix.GetInverse();

    real32 NdcX = (ScreenPoint.X / (real32)BackBufferW) * 2.f - 1.f;
    real32 NdcY = (((real32)BackBufferH - ScreenPoint.Y) / (real32)BackBufferH) * 2.f - 1.f;
    real32 NdcZ = 0.f;

    // NOTE(Kevin): For DX11, forward is Z = +1 in NDC
    vec4 ClipSpaceRay = vec4(NdcX, NdcY, 1.f, 1.f);
    vec4 ViewSpaceRay = InvPerspectiveMatrix * ClipSpaceRay;
    ViewSpaceRay = vec4(ViewSpaceRay.x, ViewSpaceRay.y, -1.f, 0.f);
    vec3 WorldSpaceRay = Normalize((InvViewMatrix * ViewSpaceRay).xyz);
    *WorldRay = WorldSpaceRay;

    vec4 ClipSpaceP = vec4(NdcX, NdcY, NdcZ, 1.f);
    // NOTE(Kevin): For points, reverse the perspective divide after the inverse
    //              projection matrix transformation because it's simpler that way.
    vec4 ViewSpacePBeforePerspectiveDivide = InvPerspectiveMatrix * ClipSpaceP;
    vec4 ViewSpaceP = ViewSpacePBeforePerspectiveDivide / ViewSpacePBeforePerspectiveDivide.w;
    vec4 WorldSpaceP = InvViewMatrix * ViewSpaceP;
    *WorldPoint = WorldSpaceP.XYZ;
}

internal bool32
IntersectPlaneAndLine(vec3 PlaneP, vec3 PlaneN, vec3 LineP,
                      vec3 LineDir, vec3 *IntersectionP)
{
    real32 Denominator = Dot(PlaneN, LineDir);

    if (AbsoluteValue(Denominator) < 0.00001f)
    {
        return false;
    }

    real32 t = (Dot(PlaneN, PlaneP) - Dot(PlaneN, LineP)) / Denominator;
    *IntersectionP = LineP + LineDir * t;

    return true;
}

internal bool32
IntersectPlaneAndLineWithDirections(vec3 PlaneP, vec3 PlaneN, vec3 LineP,
                                    vec3 LineDir, vec3 *IntersectionP)
{
    real32 Denominator = Dot(PlaneN, LineDir);

    if (Denominator > -0.000001f)
    {
        return false;
    }

    real32 t = (Dot(PlaneN, PlaneP) - Dot(PlaneN, LineP)) / Denominator;
    *IntersectionP = LineP + LineDir * t;

    return true;
}

internal void
GetDrawRectBasisAxes(vec3 FloorNormal, vec3 *RectBasisUAxis, vec3 *RectBasisVAxis)
{
    vec3 FloorNormalNoY = vec3(FloorNormal.X, 0, FloorNormal.Z);

    if (Magnitude(FloorNormalNoY) < 0.001f)
    {
        FloorNormalNoY = GMATH_FORWARD;
    }
    else if (AbsoluteValue(FloorNormal.y) < 0.001f)
    {
        FloorNormalNoY = GMATH_UP;
    }
    else
    {
        FloorNormalNoY = Normalize(FloorNormalNoY);
    }

    *RectBasisUAxis = Normalize(Cross(FloorNormal, FloorNormalNoY));
    *RectBasisVAxis = Normalize(Cross(FloorNormal, *RectBasisUAxis));
}

internal void
DeleteAllFaceRenderMeshes(world_mesh *World, renderer *RenderAPI)
{
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(World->Faces); ++FaceIndex)
    {
        mesh_face *Face = &World->Faces[FaceIndex];

        if (Face->RenderMesh)
        {
            RenderAPI->DeleteMesh(Face->RenderMesh);
            Face->RenderMesh = 0;
        }
    }
}

internal void
TriangulateAndUploadFaces(world_mesh *World, memory_blocks *EditorMemory, renderer *RenderAPI)
{
    // TODO(Kevin): This should be trivial to multithread. ID3D11Device->CreateBuffer
    //              is thread-safe so I should make UploadMeshData also thread-safe
    //              maybe using atomic incrementer for retrieving handles.

    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(World->Faces); ++FaceIndex)
    {
        mesh_face *Face = &World->Faces[FaceIndex];

        if (Face->Alive)
        {
            if (!Face->RenderMesh)
            {
                // TODO(Kevin): The number of draw calls might get a bit out of hand...need to
                //              test and see if storing the vertices in memory then streaming
                //              them while coalescing faces with same texture is faster.
                vertex_buffer FaceMesh = TriangulateEditMeshFace(EditorMemory, World, FaceIndex);
                if (FaceMesh.IC > 2)
                {
                    Face->RenderMesh = RenderAPI->UploadMeshData(FaceMesh.VB, FaceMesh.VC,
                                                                 FaceMesh.IB, FaceMesh.IC);
                }
                FreeBlock(EditorMemory, FaceMesh.VB);
                FreeBlock(EditorMemory, FaceMesh.IB);
            }
        }
    }
}

internal u32
NextUndoIndex(u32 Index)
{
    u32 Result = (Index + 1) % ArrayCount(((undo_stack *)0)->WorldStates);
    return Result;
}

internal u32
PrevUndoIndex(u32 Index)
{
    u32 Result = (Index + ArrayCount(((undo_stack *)0)->WorldStates) - 1) % 
                 ArrayCount(((undo_stack *)0)->WorldStates);
    return Result;    
}

internal bool32
UndoWorldState(undo_stack *Undo, world_mesh *World, 
               memory_blocks *EditorMemory, renderer *RenderAPI)
{
    if (Undo->StackP != Undo->HistoryStartP)
    {
        DeleteAllFaceRenderMeshes(World, RenderAPI);

        Undo->StackP = PrevUndoIndex(Undo->StackP);
        Copy(World, &(Undo->WorldStates[Undo->StackP]), sizeof(world_mesh));

        TriangulateAndUploadFaces(World, EditorMemory, RenderAPI);

        return true;
    }
    else
    {
        return false;
    }
}

internal bool32
RedoWorldState(undo_stack *Undo, world_mesh *World,
               memory_blocks *EditorMemory, renderer *RenderAPI)
{
    if (Undo->StackP != Undo->MostRecentP)
    {
        DeleteAllFaceRenderMeshes(World, RenderAPI);

        Undo->StackP = NextUndoIndex(Undo->StackP);
        Copy(World, &(Undo->WorldStates[Undo->StackP]), sizeof(world_mesh));

        TriangulateAndUploadFaces(World, EditorMemory, RenderAPI);

        return true;
    }
    else
    {
        return false;
    }
}

internal void
PushWorldStateToUndoStack(undo_stack *Undo, world_mesh *World)
{
    u32 NextStackP = NextUndoIndex(Undo->StackP);

    if (NextStackP == Undo->HistoryStartP)
    {
        // If we are about to write to HistoryStartP, increment HistoryStartP
        Undo->HistoryStartP = NextUndoIndex(Undo->HistoryStartP);

        // NOTE(Kevin): Previous HistoryStartP is now unreachable
    }

    if (Undo->StackP != Undo->MostRecentP)
    {
        // NOTE(Kevin): Everything from [NextStackP, Undo->MostRecentP] will become unreachable
    }

    Undo->StackP = NextStackP;
    Undo->MostRecentP = NextStackP;

    world_mesh *PushedState = &(Undo->WorldStates[Undo->StackP]);
    Copy(PushedState, World, sizeof(world_mesh));

    // NOTE(Kevin): Rather than keeping all the hw meshes of the undo states alive
    //              (which would require lifetime management i.e. determine whether
    //              a hw mesh is unreachable; difficult if hw meshes allowed to be
    //              shared between undo states; if disallow, then with every modification
    //              I must retriangulate and upload whole new meshes), delete all hw
    //              meshes, retriangulate, and upload new hw meshes upon undo/redo.
    //              This way, I pay this cost at undo/redo time, not at modification
    //              time. Only dirty faces need to be deleted/recreated on modifications.
    //              This keeps modifications snappy, while undo/redo action takes longer.
    // NOTE(Kevin): Don't push any hw mesh handles to undo stack
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(PushedState->Faces); ++FaceIndex)
    {
        mesh_face *Face = &PushedState->Faces[FaceIndex];
        Face->RenderMesh = 0;
    }
}

internal void
SetGuiTooltipWithUndoState(gui_state *GuiState, undo_stack *Undo)
{
    s32 HowFarBack;
    if (Undo->MostRecentP >= Undo->StackP)
    {
        HowFarBack = (s32)Undo->StackP - (s32)Undo->MostRecentP;
    }
    else
    {
        HowFarBack = (s32)Undo->StackP - (s32)(Undo->MostRecentP + 
                                         ArrayCount(Undo->WorldStates));
    }

    if (HowFarBack == 0)
    {
        char *Tooltip = "Viewing world state (Most Recent)";
        GuiState->TooltipLength = StringLength(Tooltip);
        Copy(GuiState->Tooltip, Tooltip, GuiState->TooltipLength+1);
    }
    else
    {
        GuiState->TooltipLength = stbsp_sprintf(GuiState->Tooltip,
                                                "Viewing world state (%d)", HowFarBack);
    }
}

internal void
EnterNextMode(editor *Editor)
{
    // NOTE(Kevin): Handle all mode switching here!

    if (Editor->QueuedMode == NothingMode ||
        Editor->QueuedMode == Editor->ActiveMode)
    {
        return;
    }

    // Deactivate active mode
    switch (Editor->ActiveMode)
    {
        case BlockTool:
        case PolygonTool:
            break;
    }

    Editor->ActiveMode = Editor->QueuedMode;
    Editor->QueuedMode = NothingMode;

    // Activate new mode
    switch (Editor->ActiveMode)
    {
        case BlockTool:
        case PolygonTool:
            break;
    }
}

internal void
ResetGrid(editor *Editor)
{
    Editor->GridOrigin = vec3();
    Editor->GridUp = GMATH_UP;
    Editor->GridRight = GMATH_RIGHT;
}

internal mat3
GetGridRotationMatrix(editor *Editor)
{
    mat3 Result;
    Result.columns[0] = Normalize(Cross(Editor->GridUp, Editor->GridRight));
    Result.columns[1] = Normalize(Editor->GridUp);
    Result.columns[2] = Normalize(Editor->GridRight);
    return Result;
}

internal real32
SnapToGrid(editor *Editor, real32 BeforeSnap)
{
    real32 Result = RoundReal32(BeforeSnap / Editor->GridIncrement) * Editor->GridIncrement;
    return Result;
}

internal vec3
SnapToGrid(editor *Editor, vec3 BeforeSnap)
{
    vec3 SnapTemp = BeforeSnap;
    mat3 Rot = GetGridRotationMatrix(Editor);
    mat3 InvRot = Rot.GetInverse();
    SnapTemp = SnapTemp - Editor->GridOrigin;
    SnapTemp = InvRot * SnapTemp;
    SnapTemp = { SnapToGrid(Editor, SnapTemp.X),
                 SnapToGrid(Editor, SnapTemp.Y),
                 SnapToGrid(Editor, SnapTemp.Z) };
    SnapTemp = Rot * SnapTemp;
    SnapTemp = SnapTemp + Editor->GridOrigin;
    return SnapTemp;
}

internal vec3
IndexToColor(u32 Index)
{
    real32 R = (real32) ((Index & 0x000000FF) >> 0);
    real32 G = (real32) ((Index & 0x0000FF00) >> 8);
    real32 B = (real32) ((Index & 0x00FF0000) >> 16);
    return vec3(R,G,B)/255.f;
}

internal u32
ColorToIndex(u32 RGBA)
{
    u32 Index = ((RGBA & 0xFF000000) >> 24) |
                ((RGBA & 0x00FF0000) >> 8) |
                ((RGBA & 0x0000FF00) << 8);
    return Index;
}

internal edge_index
PickEdge(editor *Editor)
{
    // TODO(Kevin): Render all the edges with their edge_index as their color
    return 0;
}

internal face_index
PickFace(editor *Editor)
{
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(Editor->World->Faces); ++FaceIndex)
    {
        mesh_face *Face = &Editor->World->Faces[FaceIndex];

        if (Face->Alive)
        {
            if (Face->RenderMesh)
            {
                render_item RenderItem;
                RenderItem = {};
                RenderItem.Descriptor.Mesh = Face->RenderMesh;
                RenderItem.Descriptor.Albedo = WHITE_TEXTURE_1X1;
                RenderItem.Descriptor.Lightmap = WHITE_TEXTURE_1X1;
                RenderItem.UseLightmap = true;
                RenderItem.UseLightCubes = false;
                RenderItem.ForcedAmbient = IndexToColor(FaceIndex);
                RenderItem.WorldPosition = vec3();
                RenderItem.WorldRotation = quat();
                RenderItem.WorldScale = vec3(1,1,1);
                Editor->RenderAPI->PushRenderItems(&RenderItem, 1);
            }
        }
    }

    u32 Pixel = Editor->RenderAPI->FlushAndReadbackPixel(FloorReal32ToU32(Editor->ViewportMouseP.X),
                                                         FloorReal32ToU32(Editor->ViewportMouseP.Y));
    u32 Index = ColorToIndex(Pixel);

    return Index;
}

internal bool32
PickPointUnderMouse(editor *Editor, vec3 *PlanePoint, vec3 *PlaneNormal)
{
    face_index PickedFace = Editor->FaceUnderMouse;
    if (PickedFace == 0)
    {
        PickedFace = PickFace(Editor);
    }

    if (PickedFace > 0)
    {
        *PlaneNormal = GetFaceNormal(Editor->World, PickedFace);

        mesh_face DrawingFace = Editor->World->Faces[PickedFace];

        vec3 WP, WR;
        ScreenPointToWorldPointAndRay(Editor->ViewportMouseP, Editor->RenderAPI, &WP, &WR);

        vert_index VertOnFace = Editor->World->Loops[DrawingFace.LoopCycleBase].Vert;
        IntersectPlaneAndLineWithDirections(Editor->World->Vertices[VertOnFace].P,
                                            *PlaneNormal, WP, WR, PlanePoint);
    }
    else
    {
        vec3 WP, WR;
        ScreenPointToWorldPointAndRay(Editor->ViewportMouseP, Editor->RenderAPI, &WP, &WR);

        real32 f = (0.f - WP.Y) / WR.Y;
        if (f < 0.f)
        {
            return false;
        }

        *PlanePoint = WP + WR * f;
        *PlaneNormal = { 0, 1, 0 };
    }

    return true;
}

internal void
MakeEditableRectangle(world_mesh *World, vec3 RectStartPoint, vec3 RectEndPoint,
                      vec3 FloorNormal)
{
    vec3 RectBasisUAxis, RectBasisVAxis;
    GetDrawRectBasisAxes(FloorNormal, &RectBasisUAxis, &RectBasisVAxis);
    vec3 AlongZAxis = Dot(RectEndPoint - RectStartPoint, RectBasisUAxis) * RectBasisUAxis;

    vec3 FloorPointA = RectStartPoint;
    vec3 FloorPointB = RectStartPoint + AlongZAxis;
    vec3 FloorPointC = RectEndPoint;
    vec3 FloorPointD = RectEndPoint - AlongZAxis;

#if 1
    vert_index V0 = CreateVert(World, FloorPointA);
    vert_index V1 = CreateVert(World, FloorPointB);
    vert_index V2 = CreateVert(World, FloorPointC);
    vert_index V3 = CreateVert(World, FloorPointD);

    edge_index E0 = CreateEdge(World, V0, V1);
    edge_index E1 = CreateEdge(World, V1, V2);
    edge_index E2 = CreateEdge(World, V2, V3);
    edge_index E3 = CreateEdge(World, V3, V0);

    edge_index Edges[] = { E0, E1, E2, E3 };
#else
    // Ear clipping test!
    vert_index V0 = CreateVert(World, FloorPointA);
    vert_index V1 = CreateVert(World, FloorPointB);
    vert_index V2 = CreateVert(World, FloorPointC);
    vert_index V3 = CreateVert(World, 0.2f*(RectEndPoint - RectStartPoint) + RectStartPoint);
    vert_index V4 = CreateVert(World, FloorPointD);

    edge_index E0 = CreateEdge(World, V0, V1);
    edge_index E1 = CreateEdge(World, V1, V2);
    edge_index E2 = CreateEdge(World, V2, V3);
    edge_index E3 = CreateEdge(World, V3, V4);
    edge_index E4 = CreateEdge(World, V4, V0);

    edge_index Edges[] = { E0, E1, E2, E3, E4 };
#endif

    CreateFace(World, Edges, ArrayCount(Edges));
}

internal void
DoBlockTool(editor *Editor)
{
    // TODO(Kevin): Can't use block tool if about to exceed face/elements limit!

    if (Editor->BlockToolStage == 0)
    {
        if (Editor->ViewportLMBWentDown)
        {
            bool32 ValidPick = PickPointUnderMouse(Editor, &Editor->RectStartP,
                                                   &Editor->RectDrawPlaneN);

            if (ValidPick)
            {
                Editor->RectStartP = SnapToGrid(Editor, Editor->RectStartP);

                GetDrawRectBasisAxes(Editor->RectDrawPlaneN, &Editor->RectDrawBasisUAxis,
                                     &Editor->RectDrawBasisVAxis);

                Editor->BlockToolStage = 1;
            }
        }
    }
    else if (Editor->BlockToolStage == 1)
    {
        Editor->GridOrigin = Editor->RectStartP;
        Editor->GridRight = Editor->RectDrawBasisUAxis;
        Editor->GridUp = Editor->RectDrawPlaneN;

        if (Editor->ViewportLMBIsDown || Editor->ViewportLMBWentUp)
        {
            vec3 WP, WR;
            ScreenPointToWorldPointAndRay(Editor->ViewportMouseP, Editor->RenderAPI, &WP, &WR);

            vec3 Intersection;
            IntersectPlaneAndLineWithDirections(Editor->RectStartP, Editor->RectDrawPlaneN, 
                                                WP, WR, &Intersection);

            Editor->RectEndP = SnapToGrid(Editor, Intersection);
            vec3 StartToEnd = vec3(Editor->RectEndP - Editor->RectStartP);
            vec3 StartToEndAlongBasisU = Dot(StartToEnd, Editor->RectDrawBasisUAxis) *
                                         Editor->RectDrawBasisUAxis;
            vec3 StartToEndAlongBasisV = Dot(StartToEnd, Editor->RectDrawBasisVAxis) *
                                         Editor->RectDrawBasisVAxis;

            if (Editor->RectStartP == Editor->RectEndP)
            {
                // TODO(Kevin): PushDot or PushDisc
                Editor->RenderAPI->PushLine(Editor->RectStartP-vec3(0.3f,0,0),
                                            Editor->RectEndP  +vec3(0.3f,0,0),
                                            vec3(RGBHEXTO1(0xff8000)), 2.f);
            }
            else
            {
                Editor->RenderAPI->PushLine(Editor->RectStartP, Editor->RectStartP + StartToEndAlongBasisV,
                                            vec3(RGBHEXTO1(0xff8000)), 2.f);
                Editor->RenderAPI->PushLine(Editor->RectStartP, Editor->RectStartP + StartToEndAlongBasisU,
                                            vec3(RGBHEXTO1(0xff8000)), 2.f);
                Editor->RenderAPI->PushLine(Editor->RectEndP, Editor->RectEndP - StartToEndAlongBasisV,
                                            vec3(RGBHEXTO1(0xff8000)), 2.f);
                Editor->RenderAPI->PushLine(Editor->RectEndP, Editor->RectEndP - StartToEndAlongBasisU,
                                            vec3(RGBHEXTO1(0xff8000)), 2.f);
            }

            if (Editor->ViewportLMBWentUp)
            {
                if (Magnitude(StartToEndAlongBasisU)+0.05f >= Editor->GridIncrement &&
                    Magnitude(StartToEndAlongBasisV)+0.05f >= Editor->GridIncrement)
                {
                    Editor->BlockToolStage = 2;
                    Editor->BlockDrawHeight = {};
                }
                else
                {
                    Editor->BlockToolStage = 0;
                }                
            }
        }

        // TODO(Kevin): 
        // if (AppState->KeysPressed[SDL_SCANCODE_ESCAPE])
        // {
        //     simpleBrushToolState = SimpleBrushToolState::NotActive;
        // }
    }
    else if (Editor->BlockToolStage == 2)
    {
        vec3 StartToEnd = vec3(Editor->RectEndP - Editor->RectStartP);
        real32 UComponent = Dot(StartToEnd, Editor->RectDrawBasisUAxis);
        real32 VComponent = Dot(StartToEnd, Editor->RectDrawBasisVAxis);
        vec3 StartToEndAlongBasisU = UComponent * Editor->RectDrawBasisUAxis;
        vec3 StartToEndAlongBasisV = VComponent * Editor->RectDrawBasisVAxis;

        Editor->GridOrigin = Editor->RectEndP;
        Editor->GridUp = StartToEndAlongBasisU;
        Editor->GridRight = StartToEndAlongBasisV;

        // NOTE(Kevin): Setting the height. Let plane be defined at RectEndP
        //              with normal -CameraDirection. Then the point of intersection
        //              between mouse ray and the plane - RectEndP and project it onto
        //              the direction of translation e.g. GMATH_UP
        vec3 DrawingSurfaceN = Editor->RectDrawPlaneN;
        vec3 PlaneN = -Editor->CameraDirection;
        vec3 PlaneP = Editor->RectEndP + Editor->BlockDrawHeight;

        vec3 WP, WR;
        ScreenPointToWorldPointAndRay(Editor->ViewportMouseP, Editor->RenderAPI, &WP, &WR);

        vec3 Intersection;
        IntersectPlaneAndLineWithDirections(PlaneP, PlaneN, WP, WR, &Intersection);

        real32 TrueHeightComp = Dot((Intersection - Editor->RectEndP), DrawingSurfaceN);
        Editor->BlockDrawHeight = TrueHeightComp * DrawingSurfaceN;

        real32 HeightComp = SnapToGrid(Editor, TrueHeightComp);
        vec3 Height = HeightComp * DrawingSurfaceN;

        vec3 FloorPointA = Editor->RectStartP;
        vec3 FloorPointB = Editor->RectStartP + StartToEndAlongBasisV;
        vec3 FloorPointC = Editor->RectEndP;
        vec3 FloorPointD = Editor->RectEndP - StartToEndAlongBasisV;
        // given how i've set up points ABCD:
        //      if VComponent < 0, swap A and B, swap C and D
        //      if UComponent < 0, swap B and C, swap A and D
        if (VComponent < 0)
        {
            vec3 Temp = FloorPointA;
            FloorPointA = FloorPointB;
            FloorPointB = Temp;
            Temp = FloorPointC;
            FloorPointC = FloorPointD;
            FloorPointD = Temp;
        }
        if (UComponent < 0)
        {
            vec3 Temp = FloorPointB;
            FloorPointB = FloorPointC;
            FloorPointC = Temp;
            Temp = FloorPointA;
            FloorPointA = FloorPointD;
            FloorPointD = Temp;
        }
        vec3 CeilPointA = FloorPointA + Height;
        vec3 CeilPointB = FloorPointB + Height;
        vec3 CeilPointC = FloorPointC + Height;
        vec3 CeilPointD = FloorPointD + Height;

        Editor->RenderAPI->PushLine(FloorPointA, FloorPointB, vec3(RGBHEXTO1(0xff8000)), 2.f);
        Editor->RenderAPI->PushLine(FloorPointA, FloorPointD, vec3(RGBHEXTO1(0xff8000)), 2.f);
        Editor->RenderAPI->PushLine(FloorPointC, FloorPointB, vec3(RGBHEXTO1(0xff8000)), 2.f);
        Editor->RenderAPI->PushLine(FloorPointC, FloorPointD, vec3(RGBHEXTO1(0xff8000)), 2.f);
        Editor->RenderAPI->PushLine(FloorPointA, CeilPointA, vec3(RGBHEXTO1(0xff8000)), 2.f);
        Editor->RenderAPI->PushLine(FloorPointB, CeilPointB, vec3(RGBHEXTO1(0xff8000)), 2.f);
        Editor->RenderAPI->PushLine(FloorPointC, CeilPointC, vec3(RGBHEXTO1(0xff8000)), 2.f);
        Editor->RenderAPI->PushLine(FloorPointD, CeilPointD, vec3(RGBHEXTO1(0xff8000)), 2.f);
        Editor->RenderAPI->PushLine(CeilPointA, CeilPointB, vec3(RGBHEXTO1(0xff8000)), 2.f);
        Editor->RenderAPI->PushLine(CeilPointA, CeilPointD, vec3(RGBHEXTO1(0xff8000)), 2.f);
        Editor->RenderAPI->PushLine(CeilPointC, CeilPointB, vec3(RGBHEXTO1(0xff8000)), 2.f);
        Editor->RenderAPI->PushLine(CeilPointC, CeilPointD, vec3(RGBHEXTO1(0xff8000)), 2.f);

        if (Editor->ViewportLMBWentUp)
        {
            // TODO(Kevin): If height component is zero or too small then
            //              just default to current grid increment

            vert_index fv0 = CreateVert(Editor->World, FloorPointA);
            vert_index fv1 = CreateVert(Editor->World, FloorPointB);
            vert_index fv2 = CreateVert(Editor->World, FloorPointC);
            vert_index fv3 = CreateVert(Editor->World, FloorPointD);
            vert_index cv0 = CreateVert(Editor->World, CeilPointA);
            vert_index cv1 = CreateVert(Editor->World, CeilPointB);
            vert_index cv2 = CreateVert(Editor->World, CeilPointC);
            vert_index cv3 = CreateVert(Editor->World, CeilPointD);

            if (HeightComp < 0)
            {
                vert_index Temp0 = fv0;
                vert_index Temp1 = fv1;
                vert_index Temp2 = fv2;
                vert_index Temp3 = fv3;
                fv0 = cv0;
                fv1 = cv1;
                fv2 = cv2;
                fv3 = cv3;
                cv0 = Temp0;
                cv1 = Temp1;
                cv2 = Temp2;
                cv3 = Temp3;
            }

            edge_index f0_to_f1 = CreateEdge(Editor->World, fv0, fv1);
            edge_index f1_to_f2 = CreateEdge(Editor->World, fv1, fv2);
            edge_index f2_to_f3 = CreateEdge(Editor->World, fv2, fv3);
            edge_index f3_to_f0 = CreateEdge(Editor->World, fv3, fv0);
            edge_index c0_to_c1 = CreateEdge(Editor->World, cv0, cv1);
            edge_index c1_to_c2 = CreateEdge(Editor->World, cv1, cv2);
            edge_index c2_to_c3 = CreateEdge(Editor->World, cv2, cv3);
            edge_index c3_to_c0 = CreateEdge(Editor->World, cv3, cv0);
            edge_index f0_to_c0 = CreateEdge(Editor->World, fv0, cv0);
            edge_index f1_to_c1 = CreateEdge(Editor->World, fv1, cv1);
            edge_index f2_to_c2 = CreateEdge(Editor->World, fv2, cv2);
            edge_index f3_to_c3 = CreateEdge(Editor->World, fv3, cv3);

            edge_index FloorEdges[] = { f0_to_f1, f1_to_f2, f2_to_f3, f3_to_f0 };
            edge_index CeilEdges[] = { c0_to_c1, c1_to_c2, c2_to_c3, c3_to_c0 };
            CreateFace(Editor->World, FloorEdges, ArrayCount(FloorEdges));
            face_index ReverseThis = CreateFace(Editor->World, CeilEdges, ArrayCount(CeilEdges));
            ReverseFaceLoop(Editor->World, ReverseThis);
            edge_index Side0Edges[] = { f0_to_f1, f0_to_c0, f1_to_c1, c0_to_c1 };
            edge_index Side1Edges[] = { f1_to_f2, f1_to_c1, c1_to_c2, f2_to_c2 };
            edge_index Side2Edges[] = { f2_to_f3, f2_to_c2, c2_to_c3, f3_to_c3 };
            edge_index Side3Edges[] = { f3_to_f0, f3_to_c3, c3_to_c0, f0_to_c0 };
            CreateFace(Editor->World, Side0Edges, ArrayCount(Side0Edges));
            CreateFace(Editor->World, Side1Edges, ArrayCount(Side1Edges));
            CreateFace(Editor->World, Side2Edges, ArrayCount(Side2Edges));
            CreateFace(Editor->World, Side3Edges, ArrayCount(Side3Edges));

            Editor->BlockToolStage = 0;

            PushWorldStateToUndoStack(Editor->UndoStack, Editor->World);
        }

        // if (AppState->KeysPressed[SDL_SCANCODE_ESCAPE])
        // {
        //     simpleBrushToolState = SimpleBrushToolState::NotActive;
        // }
    }
}

internal void
EditorUpdateAndRender(thread_context *Thread, game_state *GameState, gui_state *GuiState,
                      kbm_input *Input, renderer *RenderAPI, real32 dt)
{
    editor *Editor = &(GameState->Editor);
    world_mesh *World = Editor->World;
    Editor->RenderAPI = RenderAPI;

    if (!Editor->IsInitialized)
    {
        Editor->World = AllocStruct(&Editor->EditorMemory, world_mesh);
        InitializeWorldMesh(Editor->World);
        World = Editor->World;
        World->Memory = &Editor->EditorMemory;

#if 1
        MakeEditableRectangle(World, { 100, 0, 100 }, { -100, 0, -100 }, GMATH_UP);

        edge_index ToExtrude[] = { 1, 3 };
        ExtrudeEdges(World, ToExtrude, ArrayCount(ToExtrude));
        World->Vertices[5].P += vec3(0, 32, 0);
        World->Vertices[6].P += vec3(0, 32, 0);
        World->Vertices[7].P += vec3(0, 16, 0);
        World->Vertices[8].P += vec3(0, 32, 0);

        // SplitEdgeMakeVert(World, 1, 0.3f);
        // SplitEdgeMakeVert(World, 2, 0.5f);
        // SplitEdgeMakeVert(World, 3, 0.5f);
        // SplitEdgeMakeVert(World, 4, 0.5f);
        // ReverseFaceLoop(World, 1);
#else
        for (s32 I = 1; I < 10000; ++I)
        {
            real32 Y = ((I/1000)-5)*64.f;
            real32 X = (((I%1000)/32)-16)*64.f;
            real32 Z = (((I%1000)%32)-16)*64.f;
            MakeEditableRectangle(World, { X, Y, Z }, { X+32.f, Y, Z+32.f }, GMATH_UP);
        }
#endif
        Editor->UndoStack = AllocStruct(&Editor->EditorMemory, undo_stack);
        // NOTE(Kevin): Push initial world state to undo stack
        TriangulateAndUploadFaces(World, &Editor->EditorMemory, RenderAPI);
        PushWorldStateToUndoStack(Editor->UndoStack, World);
        Editor->UndoStack->HistoryStartP = Editor->UndoStack->StackP;

        Editor->GridIncrement = (real32)STANDARD_LENGTH_IN_GAME_UNITS;

        // TODO(Kevin): For face colliders, I should first detect whether the
        // face polygon is convex or concave: if concave, then I may need to
        // treat each triangle as a collider; if convex, then I'm fine to treat
        // the whole polygon as a collider.

        Editor->IsInitialized = true;
    }

    Editor->ViewportMouseP = { (real32)Input->MouseX, (real32)Input->MouseY };
    Editor->ViewportLMBWentDown = false;
    if (!GuiState->Hovered)
    {
        Editor->ViewportLMBWentDown = Input->LMBWentDown;
    }
    Editor->ViewportLMBWentUp = (Editor->ViewportLMBIsDown || Editor->ViewportLMBWentDown) &&
                                Input->LMBWentUp;
    Editor->ViewportLMBIsDown = (Editor->ViewportLMBIsDown || Editor->ViewportLMBWentDown) &&
                                !Input->LMBWentUp;
    if (Editor->ViewportLMBIsDown || 
        Editor->ViewportLMBWentDown ||
        Editor->BlockToolStage > 0)
    {
        // NOTE(Kevin): UI should not be interactible while performing editor action
        GuiState->Input.MouseX = -1000;
        GuiState->Input.MouseY = -1000;
    }

    Editor->CameraDirection = GameState->Camera.ViewForward;
    EnterNextMode(Editor);
    ResetGrid(Editor);

    if (Input->KBController.ActionUp.EndedDown && 
        Input->KBController.ActionUp.HalfTransitionCount > 0)
    {
        for (u32 VertIndex = 0; VertIndex < ArrayCount(World->Vertices); ++VertIndex)
        {
            mesh_vert *Vert = &World->Vertices[VertIndex];
            Vert->P += vec3(0, 16, 0);
        }

        // TODO(Kevin): Only need to delete+recreate the dirty faces, but right now its all dirty
        DeleteAllFaceRenderMeshes(World, RenderAPI);
        TriangulateAndUploadFaces(World, &Editor->EditorMemory, RenderAPI);

        // NOTE(Kevin): Create snapshot of world after changes/modification
        PushWorldStateToUndoStack(Editor->UndoStack, World);
        GuiState->TooltipTimer = 0.f;
    }

    if (Input->KBController.ActionLeft.EndedDown &&
        Input->KBController.ActionLeft.HalfTransitionCount > 0)
    {
        bool32 ValidUndo = UndoWorldState(Editor->UndoStack, World,
                                          &Editor->EditorMemory, RenderAPI);
        SetGuiTooltipWithUndoState(GuiState, Editor->UndoStack);
        GuiState->TooltipTimer = 1.f;

        // NOTE(Kevin): Reset tool sub-modes
        Editor->BlockToolStage = 0;
    }

    if (Input->KBController.ActionRight.EndedDown &&
        Input->KBController.ActionRight.HalfTransitionCount > 0)
    {
        bool32 ValidRedo = RedoWorldState(Editor->UndoStack, World,
                                          &Editor->EditorMemory, RenderAPI);
        SetGuiTooltipWithUndoState(GuiState, Editor->UndoStack);
        GuiState->TooltipTimer = 1.f;
    }

    Editor->EdgeUnderMouse = 0;
    Editor->FaceUnderMouse = 0;
    if (!Input->MouseCaptured && !GuiState->Hovered)
    {
        switch (Editor->ActiveMode)
        {
            case SelectEdge:
                Editor->EdgeUnderMouse = PickEdge(Editor);
                break;
            case SelectFace:
                Editor->FaceUnderMouse = PickFace(Editor);
                break;
            case BlockTool:
            default:
                break;
        }
    }

    switch (Editor->ActiveMode)
    {
        // case PLACE_POINT_ENTITY:
        //     DoPlacePointEntity();
        //     break;
        // case MOVE_POINT_ENTITY:
        //     DoMovePointEntity();
        //     break;
        case SelectFace:
            break;
        case BlockTool:
            DoBlockTool(Editor);
            break;
        case NothingMode:
        default:
            break;
    }

    TriangulateAndUploadFaces(World, &Editor->EditorMemory, RenderAPI);

    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(World->Faces); ++FaceIndex)
    {
        mesh_face *Face = &World->Faces[FaceIndex];

        if (Face->Alive)
        {
            if (Face->RenderMesh)
            {
                render_item RenderItem;
                RenderItem = {};
                RenderItem.Descriptor.Mesh = Face->RenderMesh;
                RenderItem.Descriptor.Albedo = WHITE_TEXTURE_1X1;
                RenderItem.Descriptor.Lightmap = WHITE_TEXTURE_1X1;
                RenderItem.UseLightmap = true;
                RenderItem.UseLightCubes = false;
                // TODO(Kevin): Turn off backface culling only for the hovered face...
                RenderItem.ForcedAmbient = FaceIndex == Editor->FaceUnderMouse ?
                                           vec3(1,0,0) :
                                           GetDebugColor3(FaceIndex);
                RenderItem.WorldPosition = vec3();
                RenderItem.WorldRotation = quat();
                RenderItem.WorldScale = vec3(1,1,1);
                RenderAPI->PushRenderItems(&RenderItem, 1);
            }
        }
    }

    for (u32 EdgeIndex = 0; EdgeIndex < ArrayCount(World->Edges); ++EdgeIndex)
    {
        mesh_edge *Edge = &World->Edges[EdgeIndex];

        if (Edge->Alive)
        {
            mesh_vert V0 = World->Vertices[Edge->A];
            mesh_vert V1 = World->Vertices[Edge->B];
            vec3 P0 = V0.P;
            vec3 P1 = V1.P;
            vec3 LineColor = { 0, 1, 1 };
            RenderAPI->PushLine(P0, P1, LineColor, 3.f);
        }
    }

    if (GuiState->CanvasW > 0 && GuiState->CanvasW < UINT32_MAX && 
        GuiState->CanvasH > 0 && GuiState->CanvasH < UINT32_MAX)
    {
        window_state Window = BeginWindow(GuiState, { 10, 80, 400, 600 });

        if (WindowButton(&Window, "Face Select"))
        {
            Editor->QueuedMode = SelectFace;
        }
        if (WindowButton(&Window, "Block Tool"))
        {
            Editor->QueuedMode = BlockTool;
        }

        EndWindow(&Window);

#if 0
        GuiPanel(GuiState, { GameState->Editor.EditorMouseX-5, 
                 GameState->Editor.EditorMouseY-5, 10, 10 }, 0);
#endif
#if 0
        if (GameState->Editor.IsInitialized)
        {
            GuiTextFmt(GuiState, 500, 50, GuiFontCambria40, 1.4f, Align_Left, false, vec4(1,1,1,1),
                       "SP: %d", GameState->Editor.UndoStack->StackP);
            GuiTextFmt(GuiState, 500, 100, GuiFontCambria40, 1.4f, Align_Left, false, vec4(1,1,1,1),
                       "HS: %d", GameState->Editor.UndoStack->HistoryStartP);
            GuiTextFmt(GuiState, 500, 150, GuiFontCambria40, 1.4f, Align_Left, false, vec4(1,1,1,1),
                       "MR: %d", GameState->Editor.UndoStack->MostRecentP);
        }
#endif

        if (GuiState->TooltipTimer > 0.f)
        {
            GuiState->TooltipTimer -= dt;

            if (GuiState->TooltipLength)
            {
                real32 TextWidth, _;
                real32 TextScale = 2.0f;
                VT_GetTextBoundingBoxDimensions(GuiState->Fonts[GuiFontCambria40].FontInfo, 
                                                GuiState->Tooltip, TextScale, &TextWidth, &_,
                                                GuiState->TooltipLength);
                s32 X = (s32)(GuiState->CanvasW - RoundReal32ToUInt32(TextWidth)) / 2;
                GuiText(GuiState, X, (s32)GuiState->CanvasH - 100, GuiFontCambria40, TextScale,
                        Align_Left, false, { 1.f, 1.f, 1.f, 1.f }, GuiState->Tooltip);
            }
        }
    }
}

