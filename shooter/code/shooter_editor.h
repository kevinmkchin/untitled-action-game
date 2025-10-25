#ifndef SHOOTER_EDITOR_H
#define SHOOTER_EDITOR_H


void EditorUpdateAndRender(thread_context *Thread, game_state *GameState, gui_state *GuiState,
                           game_input *Input, struct renderer *RendererAPI);

#endif // SHOOTER_EDITOR_H
