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
DeleteFaceRenderMesh(world_mesh *World, face_index FaceIndex, renderer *RenderAPI)
{
    mesh_face *Face = &World->Faces[FaceIndex];

    if (Face->RenderMesh)
    {
        RenderAPI->DeleteMesh(Face->RenderMesh);
        Face->RenderMesh = 0;
    }
}

internal void
DeleteAllFaceRenderMeshes(world_mesh *World, renderer *RenderAPI)
{
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(World->Faces); ++FaceIndex)
    {
        DeleteFaceRenderMesh(World, FaceIndex, RenderAPI);
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
            if (Face->Dirty)
            {
                DeleteFaceRenderMesh(World, FaceIndex, RenderAPI);
                Face->Dirty = false;
            }

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
                FreeBlock(EditorMemory, &FaceMesh.VB);
                FreeBlock(EditorMemory, &FaceMesh.IB);
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
UndoWorldState(editor *Editor, undo_stack *Undo, world_mesh *World, 
               memory_blocks *EditorMemory, renderer *RenderAPI)
{
    if (Undo->StackP != Undo->HistoryStartP)
    {
        DeleteAllFaceRenderMeshes(World, RenderAPI);

        Undo->StackP = PrevUndoIndex(Undo->StackP);

        undo_state *State = &(Undo->WorldStates[Undo->StackP]);
        Copy(World, &State->World, sizeof(world_mesh));
        Editor->QueuedMode = State->ActiveMode;
        EnterNextMode(Editor);

        TriangulateAndUploadFaces(World, EditorMemory, RenderAPI);

        return true;
    }
    else
    {
        return false;
    }
}

internal bool32
RedoWorldState(editor *Editor, undo_stack *Undo, world_mesh *World,
               memory_blocks *EditorMemory, renderer *RenderAPI)
{
    if (Undo->StackP != Undo->MostRecentP)
    {
        DeleteAllFaceRenderMeshes(World, RenderAPI);

        Undo->StackP = NextUndoIndex(Undo->StackP);

        undo_state *State = &(Undo->WorldStates[Undo->StackP]);
        Copy(World, &State->World, sizeof(world_mesh));
        Editor->QueuedMode = State->ActiveMode;
        EnterNextMode(Editor);

        TriangulateAndUploadFaces(World, EditorMemory, RenderAPI);

        return true;
    }
    else
    {
        return false;
    }
}

internal void
PushWorldStateToUndoStack(undo_stack *Undo, world_mesh *World, editor_mode ActiveMode)
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

    undo_state *PushedState = &(Undo->WorldStates[Undo->StackP]);
    Copy(&PushedState->World, World, sizeof(world_mesh));
    PushedState->ActiveMode = ActiveMode;

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
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(PushedState->World.Faces); ++FaceIndex)
    {
        mesh_face *Face = &PushedState->World.Faces[FaceIndex];
        Face->RenderMesh = 0;
    }
}

internal void
PopUndoStack(undo_stack *Undo)
{
    if (Undo->StackP != Undo->HistoryStartP)
    {
        Undo->StackP = PrevUndoIndex(Undo->StackP);
        Undo->MostRecentP = Undo->StackP;
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

internal s32
IsElementSelected(element_selection *Selection, u32 ElementID)
{
    s32 FoundIndex = -1;

    for (u32 Index = 0; Index < Selection->SelectionCount; ++Index)
    {
        if (Selection->Selection[Index] == ElementID)
        {
            FoundIndex = Index;
            break;
        }
    }

    return FoundIndex;
}

internal void
ClearSelection(element_selection *Selection)
{
    Selection->SelectionCount = 0;
}

internal void
ClearAllSelection(editor *Editor)
{
    ClearSelection(&Editor->FaceSelection);
    ClearSelection(&Editor->EdgeSelection);
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
        case SelectEdge:
        {
            if (Editor->QueuedMode != TranslateEdgesTool)
            {
                ClearSelection(&Editor->EdgeSelection);
            }
        } break;

        case SelectFace:
        {
            if (Editor->QueuedMode != TranslateEdgesTool)
            {
                ClearSelection(&Editor->FaceSelection);
            }
        } break;

        case TranslateEdgesTool:
        {
            if (Editor->MoveVertexOldP.List)
            {
                FreeListOfVec3(&Editor->MoveVertexOldP);
            }
            if (Editor->WorkingSetVerts.List)
            {
                FreeListOfIndices(&Editor->WorkingSetVerts);
            }
            if (Editor->WorkingSetEdges.List)
            {
                FreeListOfIndices(&Editor->WorkingSetEdges);
            }
            if (Editor->WorkingSetFaces.List)
            {
                FreeListOfIndices(&Editor->WorkingSetFaces);
            }
        } break;

        case BlockTool:
        {
            Editor->BlockToolStage = 0;
        } break;
    }

    Editor->PrevMode = Editor->ActiveMode;
    Editor->ActiveMode = Editor->QueuedMode;
    Editor->QueuedMode = NothingMode;

    // Activate new mode
    switch (Editor->ActiveMode)
    {
        case TranslateEdgesTool:
        {
            Assert(Editor->WorkingSetVerts.Count > 0);
            AllocListOfVec3(&Editor->EditorMemory, &Editor->MoveVertexOldP, Editor->WorkingSetVerts.Count);

            for (u32 I = 0; I < Editor->WorkingSetVerts.Count; ++I)
            {
                vert_index V = Editor->WorkingSetVerts.List[I];
                mesh_vert WV = Editor->World->Vertices[V];
                Editor->MoveVertexOldP.List[Editor->MoveVertexOldP.Count++] = WV.P;
            }

            vec3 WP, WR;
            ScreenPointToWorldPointAndRay(Editor->ViewportMouseP, Editor->RenderAPI, &WP, &WR);
            IntersectPlaneAndLineWithDirections(Editor->MoveVertexOldP.List[0], -Editor->CameraDirection,
                                                WP, WR, &Editor->MouseInteractStartP);
        } break;

        case NothingMode:
            break;
    }
}

internal void
DrawEdge(editor *Editor, edge_index EdgeIndex, vec4 Color, float Thickness)
{
    mesh_edge *Edge = &Editor->World->Edges[EdgeIndex];

    if (Edge->Alive)
    {
        mesh_vert V0 = Editor->World->Vertices[Edge->A];
        mesh_vert V1 = Editor->World->Vertices[Edge->B];
        vec3 P0 = V0.P;
        vec3 P1 = V1.P;
        Editor->RenderAPI->PushLine(P0, P1, Color, Thickness);
    }
}

internal void
DrawFace(editor *Editor, face_index FaceIndex, vec3 Color)
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
            RenderItem.ForcedAmbient = Color;
            RenderItem.WorldPosition = vec3();
            RenderItem.WorldRotation = quat();
            RenderItem.WorldScale = vec3(1,1,1);
            Editor->RenderAPI->PushRenderItems(&RenderItem, 1);
        }
    }
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

internal u32
RenderAndPickIDUnderMouse(editor *Editor)
{
    u32 Pixel = Editor->RenderAPI->FlushAndReadbackPixel(FloorReal32ToU32(Editor->ViewportMouseP.X),
                                                         FloorReal32ToU32(Editor->ViewportMouseP.Y));
    u32 Index = ColorToIndex(Pixel);

    return Index;
}

internal edge_index
PickEdge(editor *Editor)
{
    for (u32 EdgeIndex = 0; EdgeIndex < ArrayCount(Editor->World->Edges); ++EdgeIndex)
    {
        // TODO(Kevin): Perhaps draw a "CPU-stroked" quad per topo_loop (get px consistent
        //              size with GetEditorHandleSize). Then, we can pick edges that are
        //              overlapping. Highlight the quads of the radial cycle of this edge.
        vec4 ColorID = vec4(IndexToColor(EdgeIndex), 1.0f);
        DrawEdge(Editor, EdgeIndex, ColorID, 50.f);
    }

    u32 Index = RenderAndPickIDUnderMouse(Editor);

    return Index;
}

internal face_index
PickFace(editor *Editor)
{
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(Editor->World->Faces); ++FaceIndex)
    {
        vec3 ColorID = IndexToColor(FaceIndex);
        DrawFace(Editor, FaceIndex, ColorID);
    }

    u32 Index = RenderAndPickIDUnderMouse(Editor);

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
        IntersectPlaneAndLine(Editor->World->Vertices[VertOnFace].P,
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
TranslateWorkingSetVertices(editor *Editor, gui_state *GuiState)
{
    // TODO(Kevin): If press flip, then flip faces

    if (Editor->MouseNotCaptured)
    {
        bool32 DoVerticalTranslate = Editor->KeyboardState->KbState[KB_ALT].EndedDown;

        vec3 TotalTranslation = {};
        vec3 DragOriginP = Editor->MouseInteractStartP;

        vec3 WP, WR;
        ScreenPointToWorldPointAndRay(Editor->ViewportMouseP, Editor->RenderAPI, &WP, &WR);

        if (DoVerticalTranslate)
        {
            vec3 Intersection;
            IntersectPlaneAndLineWithDirections(DragOriginP,
                                                vec3(-Editor->CameraDirection.X, 0.f, -Editor->CameraDirection.Z), 
                                                WP, WR, &Intersection);
            real32 dY = Dot((Intersection - DragOriginP), GMATH_UP);
            TotalTranslation = { 0, dY, 0 };
            TotalTranslation = SnapToGrid(Editor, TotalTranslation);
        }
        else
        {
            vec3 Intersection;
            IntersectPlaneAndLine(DragOriginP, GMATH_UP, WP, WR, &Intersection);
            TotalTranslation = Intersection - DragOriginP;
            TotalTranslation = SnapToGrid(Editor, TotalTranslation);
        }

        if (DoVerticalTranslate)
        {
            Editor->RenderAPI->PushLine(DragOriginP, DragOriginP + TotalTranslation, vec4(0,0.8f,0,1), 1);

            GuiTextFmt(GuiState, s32(Editor->ViewportMouseP.X+3), s32(Editor->ViewportMouseP.Y+3),
                       GuiFontCambria40, 1.0f, Align_Left, false, vec4(1,1,1,1),
                       "Y: %f", TotalTranslation.Y);
        }
        else
        {
            Editor->RenderAPI->PushLine(DragOriginP+TotalTranslation, DragOriginP + vec3(0,0,TotalTranslation.Z),
                                       vec4(0.8f,0,0,1), 1);
            Editor->RenderAPI->PushLine(DragOriginP+TotalTranslation, DragOriginP + vec3(TotalTranslation.X,0,0),
                                       vec4(0,0,0.8f,1), 1);

            GuiTextFmt(GuiState, s32(Editor->ViewportMouseP.X+3), s32(Editor->ViewportMouseP.Y+3),
                       GuiFontCambria40, 1.0f, Align_Left, false, vec4(1,1,1,1),
                       "X: %f\nZ:%f", TotalTranslation.X, TotalTranslation.Z);
        }

        for (u32 I = 0; I < Editor->WorkingSetVerts.Count; ++I)
        {
            vert_index V = Editor->WorkingSetVerts.List[I];
            mesh_vert *WV = &Editor->World->Vertices[V];
            vec3 OldP = Editor->MoveVertexOldP.List[I];
            vec3 NewP = OldP + TotalTranslation;
            WV->P = NewP;
        }

        // NOTE(Kevin): Mark faces dirty and retriangulate them later
        if (Editor->WorkingSetFaces.Count > 0)
        {
            for (u32 Index = 0; Index < Editor->WorkingSetFaces.Count; ++Index)
            {
                face_index FaceIndex = Editor->WorkingSetFaces.List[Index];
                mesh_face *Face = &Editor->World->Faces[FaceIndex];
                Face->Dirty = true;
            }
        }
        else
        {
            // NOTE(Kevin): Find working set of faces
            for (u32 I = 0; I < Editor->WorkingSetVerts.Count; ++I)
            {
                vert_index V = Editor->WorkingSetVerts.List[I];
                mesh_vert *WV = &Editor->World->Vertices[V];

                for (u32 CE = 0; CE < WV->ConnectedEdgesCount; ++CE)
                {
                    edge_index CEI = WV->ConnectedEdges[CE];
                    mesh_edge *WE = &Editor->World->Edges[CEI];

                    loop_index CurrLoop = WE->RadialCycleBase;
                    for (u32 L = 0; L < WE->RadialCycleLen; ++L)
                    {
                        topo_loop Loop = Editor->World->Loops[CurrLoop];
                        mesh_face *Face = &Editor->World->Faces[Loop.Face];
                        Face->Dirty = true;
                        CurrLoop = Loop.RadialNext;
                    }
                }
            }
        }

        if (Editor->ViewportLMBWentUp)
        {
            Editor->QueuedMode = Editor->PrevMode;

            // NOTE(Kevin): Expects that we pushed the state before entering this mode,
            //              so upon completion of the mode we replace the last pushed state.
            PopUndoStack(Editor->UndoStack);
            PushWorldStateToUndoStack(Editor->UndoStack, Editor->World, Editor->PrevMode);
        }
    }
}

internal void
DoBlockTool(editor *Editor)
{
    // TODO(Kevin): Can't use block tool if about to exceed face/elements limit!
    // TODO(Kevin): Deactivate tool if ESC key pressed

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
            if (Editor->MouseNotCaptured)
            {
                vec3 WP, WR;
                ScreenPointToWorldPointAndRay(Editor->ViewportMouseP, Editor->RenderAPI, &WP, &WR);

                vec3 Intersection;
                IntersectPlaneAndLine(Editor->RectStartP, Editor->RectDrawPlaneN, 
                                      WP, WR, &Intersection);

                Editor->RectEndP = SnapToGrid(Editor, Intersection);
            }

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
                                            vec4(RGBHEXTO1(0xff8000), 1), 2.f);
            }
            else
            {
                Editor->RenderAPI->PushLine(Editor->RectStartP, Editor->RectStartP + StartToEndAlongBasisV,
                                            vec4(RGBHEXTO1(0xff8000), 1), 2.f);
                Editor->RenderAPI->PushLine(Editor->RectStartP, Editor->RectStartP + StartToEndAlongBasisU,
                                            vec4(RGBHEXTO1(0xff8000), 1), 2.f);
                Editor->RenderAPI->PushLine(Editor->RectEndP, Editor->RectEndP - StartToEndAlongBasisV,
                                            vec4(RGBHEXTO1(0xff8000), 1), 2.f);
                Editor->RenderAPI->PushLine(Editor->RectEndP, Editor->RectEndP - StartToEndAlongBasisU,
                                            vec4(RGBHEXTO1(0xff8000), 1), 2.f);
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
        IntersectPlaneAndLine(PlaneP, PlaneN, WP, WR, &Intersection);

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

        Editor->RenderAPI->PushLine(FloorPointA, FloorPointB, vec4(RGBHEXTO1(0xff8000), 1), 2.f);
        Editor->RenderAPI->PushLine(FloorPointA, FloorPointD, vec4(RGBHEXTO1(0xff8000), 1), 2.f);
        Editor->RenderAPI->PushLine(FloorPointC, FloorPointB, vec4(RGBHEXTO1(0xff8000), 1), 2.f);
        Editor->RenderAPI->PushLine(FloorPointC, FloorPointD, vec4(RGBHEXTO1(0xff8000), 1), 2.f);
        Editor->RenderAPI->PushLine(FloorPointA, CeilPointA, vec4(RGBHEXTO1(0xff8000), 1), 2.f);
        Editor->RenderAPI->PushLine(FloorPointB, CeilPointB, vec4(RGBHEXTO1(0xff8000), 1), 2.f);
        Editor->RenderAPI->PushLine(FloorPointC, CeilPointC, vec4(RGBHEXTO1(0xff8000), 1), 2.f);
        Editor->RenderAPI->PushLine(FloorPointD, CeilPointD, vec4(RGBHEXTO1(0xff8000), 1), 2.f);
        Editor->RenderAPI->PushLine(CeilPointA, CeilPointB, vec4(RGBHEXTO1(0xff8000), 1), 2.f);
        Editor->RenderAPI->PushLine(CeilPointA, CeilPointD, vec4(RGBHEXTO1(0xff8000), 1), 2.f);
        Editor->RenderAPI->PushLine(CeilPointC, CeilPointB, vec4(RGBHEXTO1(0xff8000), 1), 2.f);
        Editor->RenderAPI->PushLine(CeilPointC, CeilPointD, vec4(RGBHEXTO1(0xff8000), 1), 2.f);

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

            PushWorldStateToUndoStack(Editor->UndoStack, Editor->World, Editor->ActiveMode);
        }
    }
}

internal void
SelectElementWithMouseBehaviour(element_selection *Selection, u32 ElementID)
{
    if (ElementID > 0)
    {
        s32 FoundIndex = IsElementSelected(Selection, ElementID);

        if (FoundIndex < 0 )
        {
            if (Selection->SelectionCount < ArrayCount(Selection->Selection))
            {
                Selection->Selection[Selection->SelectionCount++] = ElementID;
            }
        }
        else
        {
            Selection->Selection[FoundIndex] = Selection->Selection[--Selection->SelectionCount];
        }
    }
    else
    {
        ClearSelection(Selection);
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

// #if 1
//         MakeEditableRectangle(World, { 100, 0, 100 }, { -100, 0, -100 }, GMATH_UP);

//         edge_index ToExtrude[] = { 1, 3 };
//         ExtrudeEdges(World, ToExtrude, ArrayCount(ToExtrude));
//         World->Vertices[5].P += vec3(0, 32, 0);
//         World->Vertices[6].P += vec3(0, 32, 0);
//         World->Vertices[7].P += vec3(0, 16, 0);
//         World->Vertices[8].P += vec3(0, 32, 0);

//         // SplitEdgeMakeVert(World, 1, 0.3f);
//         // SplitEdgeMakeVert(World, 2, 0.5f);
//         // SplitEdgeMakeVert(World, 3, 0.5f);
//         // SplitEdgeMakeVert(World, 4, 0.5f);
//         // ReverseFaceLoop(World, 1);
// #else
//         for (s32 I = 1; I < 10000; ++I)
//         {
//             real32 Y = ((I/1000)-5)*64.f;
//             real32 X = (((I%1000)/32)-16)*64.f;
//             real32 Z = (((I%1000)%32)-16)*64.f;
//             MakeEditableRectangle(World, { X, Y, Z }, { X+32.f, Y, Z+32.f }, GMATH_UP);
//         }
// #endif

        Editor->UndoStack = AllocStruct(&Editor->EditorMemory, undo_stack);
        // NOTE(Kevin): Push initial world state to undo stack
        TriangulateAndUploadFaces(World, &Editor->EditorMemory, RenderAPI);
        PushWorldStateToUndoStack(Editor->UndoStack, World, Editor->ActiveMode);
        Editor->UndoStack->HistoryStartP = Editor->UndoStack->StackP;

        Editor->GridIncrement = (real32)STANDARD_LENGTH_IN_GAME_UNITS;

        // TODO(Kevin): For face colliders, I should first detect whether the
        // face polygon is convex or concave: if concave, then I may need to
        // treat each triangle as a collider; if convex, then I'm fine to treat
        // the whole polygon as a collider.
        // 2025-12-01 I can just ear-clip triangles until I have convex polygon
        //            the clipped triangles will be their own collider.

        Editor->IsInitialized = true;
    }

    Editor->MouseOverViewport = !GuiState->Hovered;
    Editor->MouseNotCaptured = !Input->MouseCaptured;
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
        // TODO(Kevin): Find a better way
        Editor->BlockToolStage > 0 ||
        Editor->ActiveMode == TranslateEdgesTool || Editor->QueuedMode == TranslateEdgesTool)
    {
        // NOTE(Kevin): UI should not be interactible while performing editor action
        GuiState->Input.MouseX = -1000;
        GuiState->Input.MouseY = -1000;
    }
    Editor->KeyboardState = Input->KeyboardState;

    Editor->CameraDirection = GameState->Camera.ViewForward;
    Editor->WorldViewportP = GameState->Camera.ViewPosition;
    ResetGrid(Editor);

    // NOTE(Kevin): Should EnterNextMode before undo/redo because undo/redo will overwrite QueuedMode
    EnterNextMode(Editor);

    if (Input->KeyboardState->KbState[KB_CTRL].EndedDown && Input->KeyboardState->KbState[KB_SHIFT].EndedDown &&
        Input->KeyboardState->KbState[KB_Z].EndedDown && Input->KeyboardState->KbState[KB_Z].HalfTransitionCount > 0 &&
        Input->KeyboardState->KbState[KB_Z].EndedDown && Input->KeyboardState->KbState[KB_Z].HalfTransitionCount > 0)
    {
        bool32 ValidRedo = RedoWorldState(Editor, Editor->UndoStack, World,
                                          &Editor->EditorMemory, RenderAPI);

        SetGuiTooltipWithUndoState(GuiState, Editor->UndoStack);
        GuiState->TooltipTimer = 1.f;

        // TODO(Kevin): Selection should probably just be part of the undo state
        ClearAllSelection(Editor);
        // NOTE(Kevin): Reset tool sub-modes
        Editor->BlockToolStage = 0;
    }
    else if (Input->KeyboardState->KbState[KB_CTRL].EndedDown && 
             Input->KeyboardState->KbState[KB_Z].EndedDown && Input->KeyboardState->KbState[KB_Z].HalfTransitionCount > 0 &&
             Input->KeyboardState->KbState[KB_Z].EndedDown && Input->KeyboardState->KbState[KB_Z].HalfTransitionCount > 0)
    {
        bool32 ValidUndo = UndoWorldState(Editor, Editor->UndoStack, World,
                                          &Editor->EditorMemory, RenderAPI);

        SetGuiTooltipWithUndoState(GuiState, Editor->UndoStack);
        GuiState->TooltipTimer = 1.f;

        // TODO(Kevin): Selection should probably just be part of the undo state
        ClearAllSelection(Editor);
        // NOTE(Kevin): Reset tool sub-modes
        Editor->BlockToolStage = 0;
    }

    Editor->EdgeUnderMouse = 0;
    Editor->FaceUnderMouse = 0;
    if (Editor->MouseNotCaptured && Editor->MouseOverViewport)
    {
        switch (Editor->ActiveMode)
        {
            case SelectEdge:
            {
                Editor->EdgeUnderMouse = PickEdge(Editor);

                if (Editor->ViewportLMBWentDown)
                {
                    SelectElementWithMouseBehaviour(&Editor->EdgeSelection, Editor->EdgeUnderMouse);
                }
            } break;

            case SelectFace:
            {
                Editor->FaceUnderMouse = PickFace(Editor);

                if (Editor->ViewportLMBWentDown)
                {
                    SelectElementWithMouseBehaviour(&Editor->FaceSelection, Editor->FaceUnderMouse);
                }
            } break;

            default: break;
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

        case SelectEdge:
        {
            if (Input->KeyboardState->KbState[KB_G].EndedDown &&
                Input->KeyboardState->KbState[KB_G].HalfTransitionCount > 0)
            {
                if (Editor->EdgeSelection.SelectionCount > 0)
                {
                    PushWorldStateToUndoStack(Editor->UndoStack, Editor->World, Editor->ActiveMode);

                    u32 ToTranslateCount = Editor->EdgeSelection.SelectionCount;
                    AllocListOfIndices(&Editor->EditorMemory, &Editor->WorkingSetVerts, ToTranslateCount*2);

                    for (u32 Index = 0; Index < ToTranslateCount; ++Index)
                    {
                        edge_index E = Editor->EdgeSelection.Selection[Index];
                        mesh_edge Edge = Editor->World->Edges[E];
                        // TODO(Kevin): Put a bit flag in mesh_vert to indicate its already been added to working set
                        //              then unflip the bit after creating the full working set of unique vertices.
                        AddIndexIfUnique(&Editor->WorkingSetVerts.List, &Editor->WorkingSetVerts.Count, Edge.A);
                        AddIndexIfUnique(&Editor->WorkingSetVerts.List, &Editor->WorkingSetVerts.Count, Edge.B);
                    }

                    char *Tooltip = "Move edges (G)";
                    GuiState->TooltipLength = StringLength(Tooltip);
                    Copy(GuiState->Tooltip, Tooltip, GuiState->TooltipLength+1);
                    GuiState->TooltipTimer = 1.f;

                    Editor->QueuedMode = TranslateEdgesTool;
                }
            }
            else if (Input->KeyboardState->KbState[KB_E].EndedDown &&
                     Input->KeyboardState->KbState[KB_E].HalfTransitionCount > 0)
            {
                if (Editor->EdgeSelection.SelectionCount > 0)
                {
                    PushWorldStateToUndoStack(Editor->UndoStack, Editor->World, Editor->ActiveMode);

                    edge_index *ToExtrude = Editor->EdgeSelection.Selection;
                    u32 ToExtrudeCount = Editor->EdgeSelection.SelectionCount;
                    freeable_loi OppositeEdges = {};

                    ExtrudeEdges(World, ToExtrude, ToExtrudeCount,
                                 &Editor->WorkingSetVerts,
                                 &Editor->WorkingSetEdges,
                                 &Editor->WorkingSetFaces,
                                 &OppositeEdges);

                    // Select the new edges that were "cloned" from the existing edges
                    Assert(Editor->EdgeSelection.SelectionCount == OppositeEdges.Count);
                    Copy(Editor->EdgeSelection.Selection, OppositeEdges.List, sizeof(u32)*OppositeEdges.Count);
                    FreeListOfIndices(&OppositeEdges);

                    char *Tooltip = "Extrude edges (E)";
                    GuiState->TooltipLength = StringLength(Tooltip);
                    Copy(GuiState->Tooltip, Tooltip, GuiState->TooltipLength+1);
                    GuiState->TooltipTimer = 1.f;

                    Editor->QueuedMode = TranslateEdgesTool;
                }
            }
        } break;

        case SelectFace:
        {
            if (Input->KeyboardState->KbState[KB_F].EndedDown &&
                Input->KeyboardState->KbState[KB_F].HalfTransitionCount > 0)
            {
                if (Editor->FaceSelection.SelectionCount > 0)
                {
                    for (u32 Index = 0; Index < Editor->FaceSelection.SelectionCount; ++Index)
                    {
                        face_index SelectedFace = Editor->FaceSelection.Selection[Index];
                        ReverseFaceLoop(World, SelectedFace);
                        DeleteFaceRenderMesh(World, SelectedFace, RenderAPI);
                    }

                    char *Tooltip = "Flip faces (F)";
                    GuiState->TooltipLength = StringLength(Tooltip);
                    Copy(GuiState->Tooltip, Tooltip, GuiState->TooltipLength+1);
                    GuiState->TooltipTimer = 1.f;

                    PushWorldStateToUndoStack(Editor->UndoStack, World, Editor->ActiveMode);
                }
            }
            else if (Input->KeyboardState->KbState[KB_G].EndedDown &&
                     Input->KeyboardState->KbState[KB_G].HalfTransitionCount > 0)
            {
                if (Editor->FaceSelection.SelectionCount > 0)
                {
                    PushWorldStateToUndoStack(Editor->UndoStack, Editor->World, Editor->ActiveMode);

                    u32 FacesCount = Editor->FaceSelection.SelectionCount;
                    AllocListOfIndices(&Editor->EditorMemory, &Editor->WorkingSetVerts, FacesCount*4);

                    for (u32 Index = 0; Index < FacesCount; ++Index)
                    {
                        face_index F = Editor->FaceSelection.Selection[Index];
                        mesh_face Face = Editor->World->Faces[F];

                        for (loop_index l = Face.LoopCycleBase, i = 0;
                             i < Face.LoopCycleLen;
                             l = Editor->World->Loops[l].Next, ++i)
                        {
                            topo_loop Loop = Editor->World->Loops[l];
                            AddIndexIfUnique(&Editor->WorkingSetVerts.List, &Editor->WorkingSetVerts.Count, Loop.Vert);
                        }
                    }

                    char *Tooltip = "Move faces (G)";
                    GuiState->TooltipLength = StringLength(Tooltip);
                    Copy(GuiState->Tooltip, Tooltip, GuiState->TooltipLength+1);
                    GuiState->TooltipTimer = 1.f;

                    Editor->QueuedMode = TranslateEdgesTool;
                }
            }
            else if (Input->KeyboardState->KbState[KB_E].EndedDown &&
                     Input->KeyboardState->KbState[KB_E].HalfTransitionCount > 0)
            {
                // if (Editor->FaceSelection.SelectionCount > 0)
                // {
                //     PushWorldStateToUndoStack(Editor->UndoStack, Editor->World, Editor->ActiveMode);

                //     face_index *ToExtrude = Editor->FaceSelection.Selection;
                //     u32 ToExtrudeCount = Editor->FaceSelection.SelectionCount;
                //     freeable_loi OppositeEdges = {};

                //     ExtrudeEdges(World, ToExtrude, ToExtrudeCount,
                //                  &Editor->WorkingSetVerts,
                //                  &Editor->WorkingSetEdges,
                //                  &Editor->WorkingSetFaces,
                //                  &OppositeEdges);

                //     // Select the new edges that were "cloned" from the existing edges
                //     Assert(Editor->FaceSelection.SelectionCount == OppositeEdges.Count);
                //     Copy(Editor->FaceSelection.Selection, OppositeEdges.List, sizeof(u32)*OppositeEdges.Count);
                //     FreeListOfIndices(&OppositeEdges);

                //     char *Tooltip = "Extrude edges (E)";
                //     GuiState->TooltipLength = StringLength(Tooltip);
                //     Copy(GuiState->Tooltip, Tooltip, GuiState->TooltipLength+1);
                //     GuiState->TooltipTimer = 1.f;

                //     Editor->QueuedMode = TranslateEdgesTool;
                // }
            }
        } break;

        case TranslateEdgesTool:
            TranslateWorkingSetVertices(Editor, GuiState);
            break;

        case BlockTool:
            DoBlockTool(Editor);
            break;

        case NothingMode:
        default:
            break;
    }

    TriangulateAndUploadFaces(World, &Editor->EditorMemory, RenderAPI);

    // RENDER GRID
    vec3 GridP = vec3(SnapToGrid(Editor, Editor->WorldViewportP.X), 0.f,
                      SnapToGrid(Editor, Editor->WorldViewportP.Z));
    // mat3 GridR = GetGridRotationMatrix(Editor);

    RenderAPI->PushLine(vec3(320000.f, 0.f, 0.f), vec3(-320000.f, 0.f, 0.f),
                        vec4(RGB255TO1(205, 56, 9), 0.6f), 1.f);
    RenderAPI->PushLine(vec3(0.f, 320000.f, 0.f), vec3(0.f, -320000.f, 0.f),
                        vec4(RGB255TO1(67, 123, 9), 0.6f), 1.f);
    RenderAPI->PushLine(vec3(0.f, 0.f, 320000.f), vec3(0.f, 0.f, -320000.f),
                        vec4(RGB255TO1(21, 129, 205), 0.6f), 1.f);
    for (s32 I = -2000; I <= 2000; ++I)
    {
        real32 LineP = (real32)I * Editor->GridIncrement;
        real32 LineHalfLen = 32000.f;
        if (AbsoluteValue(GridP.X + LineP) > 0.01f)
        {
            RenderAPI->PushLine(GridP + vec3(LineP, 0.f, -LineHalfLen),
                                GridP + vec3(LineP, 0.f, LineHalfLen),
                                vec4(0.6f, 0.6f, 0.6f, 0.20f), 1.f);
        }
        if (AbsoluteValue(GridP.Z + LineP) > 0.01f)
        {
            RenderAPI->PushLine(GridP + vec3(-LineHalfLen, 0.f, LineP),
                                GridP + vec3(LineHalfLen, 0.f, LineP),
                                vec4(0.6f, 0.6f, 0.6f, 0.20f), 1.f);
        }
    }

    // RENDER FACES
    for (u32 FaceIndex = 0; FaceIndex < ArrayCount(World->Faces); ++FaceIndex)
    {
        // TODO(Kevin): Turn off backface culling only for the hovered face...
        vec3 Color = GetDebugColor3(FaceIndex);

        s32 FoundIndex = IsElementSelected(&Editor->FaceSelection, FaceIndex);

        if (FaceIndex == Editor->FaceUnderMouse)
        {
            Color = vec3(1, 0, 0);
        }

        if (FoundIndex >= 0)
        {
            Color = vec3(0.5, 0, 0);
        }

        DrawFace(Editor, FaceIndex, Color);
    }

    // RENDER EDGES
    for (u32 EdgeIndex = 0; EdgeIndex < ArrayCount(World->Edges); ++EdgeIndex)
    {
        vec4 Color = vec4(0, 1, 1, 1);
        real32 Thickness = 3.f;

        s32 FoundIndex = IsElementSelected(&Editor->EdgeSelection, EdgeIndex);

        if (EdgeIndex == Editor->EdgeUnderMouse)
        {
            Color = vec4(1, 0, 0, 1);
            Thickness = 8.f;
        }

        if (FoundIndex >= 0)
        {
            Color = vec4(0.5, 0, 0, 1);
            Thickness = 6.f;
        }

        DrawEdge(Editor, EdgeIndex, Color, Thickness);
    }

    if (GuiState->CanvasW > 0 && GuiState->CanvasW < UINT32_MAX && 
        GuiState->CanvasH > 0 && GuiState->CanvasH < UINT32_MAX)
    {
        window_state Window = BeginWindow(GuiState, { 60, 80, 400, 600 }, false);

        bool32 Selected = Editor->ActiveMode == SelectEdge;
        if (WindowButton(&Window, "Edge Select", &Selected))
        {
            Editor->QueuedMode = SelectEdge;
        }
        Selected = Editor->ActiveMode == SelectFace;
        if (WindowButton(&Window, "Face Select", &Selected))
        {
            Editor->QueuedMode = SelectFace;
        }
        Selected = Editor->ActiveMode == BlockTool;
        if (WindowButton(&Window, "Block Tool", &Selected))
        {
            Editor->QueuedMode = BlockTool;
        }

        EndWindow(&Window);

        GuiTextFmt(GuiState, GuiState->CanvasW - 60, 80, GuiFontCambria40, 1.0f, Align_Right, false, vec4(1,1,1,1),
                   "Selected: %d faces %d edges", Editor->FaceSelection.SelectionCount,
                   Editor->EdgeSelection.SelectionCount);

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

