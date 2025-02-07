#pragma once



enum entity_types_t
{
    POINT_PLAYER_SPAWN,
    POINT_LIGHT,
    DIRECTIONAL_LIGHT_PROPERTIES,
};

struct level_entity_t
{
    entity_types_t Type;

    // Arbitrary storage fields (can be used for any purpose)
    vec3 Position;
    vec3 Rotation; // e.g. Used as direction towards directional light source

    // each entity should have a list of key value pairs where which entries 
    // exist is defined by the entity type 
    // https://developer.valvesoftware.com/wiki/Prop_physics#Keyvalues
    // https://developer.valvesoftware.com/wiki/Generic_Keyvalues,_Inputs_and_Outputs

    void SerializeToEditableMapFile(ByteBuffer *Buf);
    void DeserializeFromEditableMapFile(ByteBuffer *Buf);
};
