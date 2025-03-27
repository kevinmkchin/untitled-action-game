
# Creating Assets for the game

## 3D Models and Characters

Use Blender for creating character models. 
- Use the +X axis in Blender as the forward direction of the character model (the character model should be facing the +X direction). When performing actions such as Symmetrize on an Armature, rotate the armature to the correct Blender forward orientation, apply, Symmetrize, rotate back to match the character model, and apply.
- Export in GLTF 2.0 binaries (so that materials are packed together with the model). Make sure to set +Y as the up axis in export settings.
- In Blender, use Principled BSDF for materials and the Image Texture used for Base Color will be read by Assimp as aiTextureType_DIFFUSE. Emissive material base color is read as aiTextureType_EMISSIVE, and either will work since I'm not doing any physically based rendering and the material is simply for importing the mesh textures.
