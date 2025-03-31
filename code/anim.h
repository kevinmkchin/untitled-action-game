/*

ANIMATION API (kevinmkchin 2025)

The resources loading code is currently only implemented for GLTF2.0 binaries. Some
other formats like DAE may load, but definitely not FBX because FBX stores nodes using
an expanded sequence of transforms because originally created for motion capture. The
animation API is format agnostic.

Multiple skinned models can share skeletons and animation clips. For example, a humanoid
skeleton and the animations for it can be shared across all humanoid characters. Of course,
the model data must be created this way first in whatever DCC e.g. Blender.

== Loading Skeletons and Animations ==

These are read-only data that should not be modified after loading.

skeleton_joint_t
skeleton_t
joint_pose_sampler_t
animation_clip_t

skeleton_t stores the joint hierarchy as a flat array. Each skeleton_joint_t contains
the inverse of its bind pose transform and the index of its parent into the flat array.
The layout of the skinning matrix palette matches exactly the layout of the joints in
this flat array. As in, if the skeleton_joint_t at index 10 has a parent index of 7,
the skinning matrix for that joint will be at position 10 in the palette and the skinning
matrix for its parent joint will be at position 7. Joints are laid out in an order that
ensures a child joint will always appear after its parent in the array i.e. joint 0 is root.

== Loading Skinned Models ==

Skinned models target a skeleton. This read-only skeleton must be loaded first.

These are read-only data that should not be modified after loading.

skinned_vertex_t
skinned_mesh_t
skinned_model_t

skinned_vertex_t stores the indices of the joints to which it is bound. A joint index is
the position of that joint in the skeleton_t joints array as well as the final skinning
matrix palette that we want to calculate.

== Per Animated Character Instance ==

animator_t

Each animated character must have an animator_t instance. The animator_t instance tracks
the animation time and stores the local/global poses of the joints in the active animation
clip.

Currently, it also stores the skinning matrix palette.


TODO
- Maybe move SkinningMatrixPalette storage out of animator_t since only one is used at a time
- Do I need "AccumulateRootJointTransform" or not?
- Use u8 instead of int for BoneID in skinned_vertex_t (also in shader and attribs)
- Change keyframe_scale_t to only use uniform-scaling and replace scale vec3 with float

*/
#pragma once

// Load just the skeleton and its animation first
bool LoadSkeleton_GLTF2Bin(const char *InFilePath, struct skeleton_t *OutSkeleton);
// Load a skinned model using a loaded skeleton
bool LoadSkinnedModel_GLTF2Bin(const char *InFilePath, struct skinned_model_t *OutSkinnedModel);
// If additional animation clips are stored in another file that uses the same skeleton
void LoadAdditionalAnimationsForSkeleton(const struct skeleton_t *Skeleton, const char *InFilePath);


#define MAX_BONES 64
#define MAX_BONE_INFLUENCE 4

struct skinned_vertex_t
{
    vec3 Pos;
    vec2 Tex;
    vec3 Norm;
    int BoneIDs[MAX_BONE_INFLUENCE]; // bone indexes which will influence this vertex
    float BoneWeights[MAX_BONE_INFLUENCE]; // weights from each bone
};

struct skinned_mesh_t
{
    u32 VAO = 0;
    u32 VBO = 0;
    u32 IBO = 0;
    u32 IndicesCount = 0;
};

struct skinned_model_t
{
    skinned_model_t(const struct skeleton_t *InSkeleton)
    {
        Skeleton = InSkeleton;
    }

    ~skinned_model_t()
    {
        LogError("skinned_model_t destructor should never be called. All animation objects are freed as part of static game memory.");
    }

    fixed_array<skinned_mesh_t> Meshes;
    fixed_array<GPUTexture> Textures;

    const struct skeleton_t *GetSkeleton() { return Skeleton; }
private:
    const struct skeleton_t *Skeleton;
};

struct skeleton_joint_t
{
    // int Id = -1; // index in FinalBonesMatrices
    mat4 InverseBindPoseTransform; // transforms vertex from model space to bone/joint space
    u8 ParentIndex;
};

struct skeleton_t
{
    mem_indexer<skeleton_joint_t> Joints;

    // look up table from joint/bone/node name to INDEX into Joints
    std::unordered_map<std::string, int> JointNameToIndex;

    c_array<struct animation_clip_t *, 32> Clips;
};


struct joint_pose_sampler_t
{
    // Extract bone keyframes from aiNodeAnim
    void Create(const struct aiNodeAnim *InChannel);
    void Delete();

    // Interpolates b/w positions, rotations, scaling keys based on the current time of 
    // the animation and prepares the local transformation matrix by combining all keys 
    // tranformations */
    mat4 SampleJointLocalPoseAt(float AnimationTime);

private:
    // Gets the index of key Positions to interpolate from based on 
    // the current animation time
    int GetPositionIndex(float AnimationTime);
    int GetRotationIndex(float AnimationTime);
    int GetScaleIndex(float AnimationTime);

    // Get normalized [0, 1] value for Lerp & Slerp
    float GetScaleFactor(float LastTimestamp, float NextTimestamp, float AnimationTime);

    // Figures out which position keys to interpolate b/w and performs
    // the interpolation and returns the translation matrix
    mat4 InterpolatePosition(float AnimationTime);
    mat4 InterpolateRotation(float AnimationTime);
    mat4 InterpolateScale(float AnimationTime);

    struct keyframe_position_t
    {
        vec3 Position;
        float Timestamp;
    };

    struct keyframe_rotation_t
    {
        quat Orientation;
        float Timestamp;
    };

    struct keyframe_scale_t
    {
        vec3 Scale;
        float Timestamp;
    };

public:
    fixed_array<keyframe_position_t> Positions;
    fixed_array<keyframe_rotation_t> Rotations;
    fixed_array<keyframe_scale_t>    Scales;
};

struct animation_clip_t
{
    float TicksPerSecond = 0.f;
    float DurationInTicks = 0.f;
    fixed_array<joint_pose_sampler_t> JointPoseSamplers;

    animation_clip_t(const skeleton_t *InSkeleton)
    {
        Skeleton = InSkeleton;
    }

    ~animation_clip_t()
    {
        LogError("animation_clip_t destructor should never be called. All animation objects are freed as part of static game memory.");
    }

    void UpdateLocalPoses(float AnimationTime, mat4 *OutLocalPoses)
    {
        ASSERT(JointPoseSamplers.length == Skeleton->Joints.count);
        for (size_t i = 0; i < JointPoseSamplers.length; ++i)
        {
            OutLocalPoses[i] = JointPoseSamplers[i].SampleJointLocalPoseAt(AnimationTime);
        }
    }

    const skeleton_t *GetSkeleton() { return Skeleton; }
private:
    const skeleton_t *Skeleton = nullptr;

};

struct animator_t
{
    animation_clip_t* CurrentAnimation = NULL;
    float CurrentTime;
    bool Looping = false;
    bool HasOwner = false;

    mat4 LocalPosesArray[MAX_BONES];
    mat4 GlobalPosesArray[MAX_BONES];
    mat4 SkinningMatrixPalette[MAX_BONES];

    void PlayAnimation(animation_clip_t *AnimationClip, bool Loop)
    {
        CurrentAnimation = AnimationClip;
        CurrentTime = 0.0f;
        Looping = Loop;
    }

    void UpdateGlobalPoses(float dt)
    {
        if (CurrentAnimation)
        {
            CurrentTime += CurrentAnimation->TicksPerSecond * dt;
            if (!Looping && CurrentTime >= CurrentAnimation->DurationInTicks)
                CurrentTime = CurrentAnimation->DurationInTicks - 0.01f;
            else
                CurrentTime = fmod(CurrentTime, CurrentAnimation->DurationInTicks);

            CurrentAnimation->UpdateLocalPoses(CurrentTime, LocalPosesArray);

            const mem_indexer<skeleton_joint_t> Joints = CurrentAnimation->GetSkeleton()->Joints;

            for (int i = 0; i < Joints.count; ++i)
            {
                u8 ParentIndex = Joints[i].ParentIndex;
                ASSERT(ParentIndex == 0xFF || ParentIndex < i);

                mat4 ParentGlobalPose = mat4(); // perhaps AccumulateRootJointTransform
                if (ParentIndex < 0xFF)
                    ParentGlobalPose = GlobalPosesArray[ParentIndex];

                GlobalPosesArray[i] = ParentGlobalPose * LocalPosesArray[i];
            }
        }
    }

    // The skinning matrix palette is passed to the rendering engine when rendering
    // a skinned mesh. For each vertex, the renderer looks up the appropriate joint’s
    // skinning matrix in the palette and uses it to transform the vertex from bind 
    // pose into current pose.
    void CalculateSkinningMatrixPalette()
    {
        if (CurrentAnimation)
        {
            const mem_indexer<skeleton_joint_t> Joints = CurrentAnimation->GetSkeleton()->Joints;
            for (int i = 0; i < Joints.count; ++i)
                SkinningMatrixPalette[i] = GlobalPosesArray[i] * Joints[i].InverseBindPoseTransform;
        }
        else
        {
            LogError("Failed to calculate skinning matrix palette as there is no animation active.");
        }
    }

};


struct ModelGLTF
{
    u16 MT_ID = 0;
    fixed_array<GPUMeshIndexed> meshes;
    fixed_array<GPUTexture> color;
};

void DrawModelInstanced(ModelGLTF& Model, int Count);
void FreeModelGLTF(ModelGLTF& model);
void RenderModelGLTF(ModelGLTF& model);
bool LoadModelGLTF2Bin(ModelGLTF *model, const char *filepath);
