
/* TODO

    I don't think we should have volumes as a concept. It's 
    unnecessary, just deal with faces. A soup of faces.

*/

struct level_editor_t
{

    void Open();
    void Close();

    void Tick();

    bool SaveMap(const char *MapFilePath);
    bool LoadMap(const char *MapFilePath);

private:
    void ResetFaceToolData();

    u32 PickVolume(MapEdit::Volume *volumes, u32 arraycount);
    u32 PickFace(MapEdit::Face **faces, u32 arraycount);

public:
    dynamic_array<MapEdit::Volume> LevelEditorVolumes;

public:
    MapEdit::Face *SelectedFace = NULL;
    db_tex_t SelectedTexture;
    // hotHandleId;
    // move all the editor session specific data here
};

