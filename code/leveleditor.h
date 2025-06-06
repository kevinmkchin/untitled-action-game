#pragma once

#include "common.h"
#include "mem.h"
#include "utility.h"
#include "winged.h"
#include "levelentities.h"
#include "cam.h"
#include "shaders.h"


enum editor_state_t
{
    NOTHING_EDITOR_STATE,

    SIMPLE_BRUSH_TOOL = 40,
    VERTEX_MANIP = 41,
    EDGE_MANIP = 42,
    FACE_MANIP = 43,

    PLACE_POINT_ENTITY,
    MOVE_POINT_ENTITY,

    INVALID_EDITOR_STATE
};

struct level_editor_t
{

    void Open();
    void Close();

    void Tick();
    void Draw();

    bool SaveMap(const char *MapFilePath);
    bool LoadMap(const char *MapFilePath);

private:
    void DoEditorGUI();

    void EnterNewStateNextFrame(editor_state_t NextState);
    void EnterNextState();

    u32 PickVolume(MapEdit::Volume *volumes, u32 arraycount);
    u32 PickFace(MapEdit::Face **faces, u32 arraycount);
    // Get the point and normal of the clicked point on a face or
    // on the XZ-plane if no face was clicked. Return false if 
    // neither were clicked.
    bool PickPointAndNormalInLevel(vec3 *PlanePoint, vec3 *PlaneNormal);
    void PickEntityBillboard();

    void DoPlacePointEntity();
    void DoMovePointEntity();
    void DoFaceManip();
    void DoVertexManip();
    void DoSimpleBrushTool();

private:
    void DrawEntityBillboards();
    void ResetFaceToolData();

public:
    dynamic_array<MapEdit::Volume> LevelEditorVolumes;
    dynamic_array<level_entity_t> LevelEntities;
    bool IsActive = false;

public:
    camera_t EditorCam;
    mat4 ActiveViewMatrix;
    mat4 ActivePerspectiveMatrix;
private:
    static constexpr float EditorNearClip = 4.f;
    static constexpr float EditorFarClip  = 8000.f;
    GPUShader Sha_EditorScene;
    GPUShader Sha_EditorWireframe;
    GPUShader Sha_EditorFaceSelected;

private: // move all the editor session specific data here
    u32 HotHandleId = 0;
    MapEdit::Face *SelectedFace = NULL;
    db_tex_t SelectedTexture;

    // Do not set directly; use EnterNewStateNextFrame and EnterNextState
    editor_state_t ActiveState = NOTHING_EDITOR_STATE;
    editor_state_t QueuedState = INVALID_EDITOR_STATE;
    editor_state_t LastState = NOTHING_EDITOR_STATE;

    bool LMBPressedThisFrame = false;
    bool LMBReleasedThisFrame = false;
    bool LMBIsPressed = false;

    int SelectedEntityIndex = -1;
    entity_types_t EntityTypeToPlace;

    std::vector<float> MY_VERTEX_BUFFER;

public:
    app_state *AppState;
    support_renderer_t *SupportRenderer;
    GPUFrameBuffer RenderTargetGame;
};

