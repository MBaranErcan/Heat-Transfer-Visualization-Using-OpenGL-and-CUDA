#include <stdio.h>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <windows.h>


#include "Kernel.cuh"
#include "Graphics.h"
#include "Shader.h"
#include "handle_error.h"

int imin(int a, int b) {
	return (a < b ? a : b);
}

int main()
{
	const float PI = 3.1415f;
	const int DIM = 512;
	const int N = (DIM * DIM);
	const float SPEED = 0.25f;
	const float MAX_TEMP = 1.0f;
	const float MIN_TEMP = 0.0001f;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((DIM + 15) / 16, (DIM + 15) / 16);


	// Init GLFW
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return -1;
	}
	// OpenGL Settings
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Init window
	GLFWwindow* window = glfwCreateWindow(DIM, DIM, "Heat Transfer Visualization Using OpenGL and CUDA", nullptr, nullptr);
	if (window == nullptr) {
		printf("Failed to create GLFW window\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Init GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cerr << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	// Init OpenGL
	glViewport(0, 0, DIM, DIM);
	glClearColor(0.5f, 0.6, 0.8f, 1.0f);

	// Create shader program
	Shader shader("Shaders/Shader.vert", "Shaders/Shader.frag");
	shader.use();
	shader.setInt("heatmap", 0); // Set the texture uniform

	// Create quad
	unsigned int VAO, VBO, EBO;
	createQuad(&VAO, &VBO, &EBO);

	// Create OpenGL PBO
	GLuint pbo;
	createPBO(&pbo, DIM);

	// Select CUDA device
	selectDevice(0);

	// Register PBO with CUDA
	DataBlock data;
	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&data.resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// Host Alloc
	float* outSrc;
	HANDLE_ERROR(cudaMallocHost((void**)&outSrc, N * sizeof(float)));

	init_anim(&data, N);

	// Fill heater
	initConstantRegion(&data, MIN_TEMP, MAX_TEMP, DIM, 256, 256, 20);

	// Main loop
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);


		animate(&data, threadsPerBlock, blocksPerGrid, SPEED, DIM);

		// Bind the PBO as a texture
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, DIM, DIM, 0, GL_RED, GL_FLOAT, nullptr);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// Bind the VAO
		glBindVertexArray(VAO);

		// OpenGL rendering
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glBindVertexArray(0);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Free memory
	anim_exit(&data);
	HANDLE_ERROR(cudaFreeHost(outSrc));
	glfwTerminate();
}