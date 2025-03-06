#pragma once

// CreateTexture
void createTexture(GLuint* texture, int width, int height) {
	glGenTextures(1, texture);
	glBindTexture(GL_TEXTURE_2D, *texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void createPBO(GLuint* pbo, int DIM) {
	glGenBuffers(1, pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, DIM * DIM * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

