#ifndef SHOOTER_RENDERER_H
#define SHOOTER_RENDERER_H

#include "shooter_types.h"

//
// NOTE(Kevin): Renderer Frontend
//
struct renderer
{
    // NOTE(Kevin): placement new to explicitly invoke the constructor
    #define ConstructRendererServiceAt(Address) new(Address) renderer()

    uint32 Test;

    // NOTE(Kevin): Service APIs to the game must be virtual to avoid linking with
    // definition. vtable essentially provides the function table of callable services
    // from the game layer. Therefore, renderer MUST be constructed to have a valid vtable
    // ptr. C-style function table will be more stable, but for now this is more convenient.
    virtual uint32 SomeTestServiceToGame(uint32 Value);
};

/*

DrawLine

*/

#endif // SHOOTER_RENDERER_H
