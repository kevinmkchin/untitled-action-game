#pragma once

/* TODO

    Still a ton to refactor in leveleditor.cpp

    I don't think we should have volumes as a concept. It's 
    unnecessary, just deal with faces. A soup of faces.

*/

struct level_editor_t
{

    void Open();
    void Close();

    void Tick();
    void Draw();

    bool SaveMap(const char *MapFilePath);
    bool LoadMap(const char *MapFilePath);

private:
    void ResetFaceToolData();

    u32 PickVolume(MapEdit::Volume *volumes, u32 arraycount);
    u32 PickFace(MapEdit::Face **faces, u32 arraycount);

public:
    dynamic_array<MapEdit::Volume> LevelEditorVolumes;
    bool IsActive = false;
public:
    vec3 CameraPosition = vec3(600, 500, 600);
    vec3 CameraRotation = vec3(0, 192.3f, 7.56f);// vec3(0,130,-30);
    vec3 CameraDirection;
    vec3 CameraRight;
    vec3 CameraUp;
    mat4 ActiveViewMatrix;
    mat4 ActivePerspectiveMatrix;
    MapEdit::Face *SelectedFace = NULL;
    db_tex_t SelectedTexture;

private:
    // hotHandleId;
    // move all the editor session specific data here

};

