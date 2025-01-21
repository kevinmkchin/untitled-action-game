#include "facebatch.h"

void CreateFaceBatch(face_batch_t *FaceBatch)
{
    glGenVertexArrays(1, &FaceBatch->VAO);
    glBindVertexArray(FaceBatch->VAO);
    glGenBuffers(1, &FaceBatch->VBO);
    glBindBuffer(GL_ARRAY_BUFFER, FaceBatch->VBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    // pos x y z
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 10, nullptr);
    glEnableVertexAttribArray(0);
    // norm i j k
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 10, (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(1);
    // uv1 u v
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 10, (void*)(sizeof(float) * (3 + 3)));
    glEnableVertexAttribArray(2);
    // uv2 u v
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 10, (void*)(sizeof(float) * (3 + 3 + 2)));
    glEnableVertexAttribArray(3);
    glBindVertexArray(0);
}

void RebindFaceBatch(face_batch_t *FaceBatch, u32 SizeInBytes, float *Data)
{
    glBindVertexArray(FaceBatch->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, FaceBatch->VBO);
    glBufferData(GL_ARRAY_BUFFER, SizeInBytes, Data, GL_DYNAMIC_DRAW);
    FaceBatch->VertexCount = SizeInBytes / (sizeof(float) * 10);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void RenderFaceBatch(const GPUShader *Shader, const face_batch_t *FaceBatch)
{
    glBindVertexArray(FaceBatch->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, FaceBatch->VBO);
    
    GLBind1i(*Shader, "ColorTexture", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, FaceBatch->ColorTexture.id);

    GLBind1i(*Shader, "LightMap", 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, FaceBatch->LightMapTexture.id);

    glDrawArrays(GL_TRIANGLES, 0, FaceBatch->VertexCount);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void DeleteFaceBatch(const face_batch_t *FaceBatch)
{
    glDeleteBuffers(1, &FaceBatch->VBO);
    glDeleteVertexArrays(1, &FaceBatch->VAO);
}
