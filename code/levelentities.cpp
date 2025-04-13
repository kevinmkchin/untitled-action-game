#include "levelentities.h"

void level_entity_t::SerializeToEditableMapFile(ByteBuffer *Buf)
{
    ByteBufferWrite(Buf, entity_types_t, Type);
    ByteBufferWrite(Buf, vec3, Position);
    ByteBufferWrite(Buf, vec3, Rotation);
}

void level_entity_t::DeserializeFromEditableMapFile(ByteBuffer *Buf)
{
    ByteBufferRead(Buf, entity_types_t, &Type);
    ByteBufferRead(Buf, vec3, &Position);
    ByteBufferRead(Buf, vec3, &Rotation);
}

