#include <stdio.h>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <windows.h>


#include "Kernel.cuh"
#include "Graphics.h"
#include "Shader.h"
#include "handle_error.h"
#include <chrono>

int imin(int a, int b) {
	return (a < b ? a : b);
}

int imax(int a, int b) {
	return (a > b ? a : b);
}

int main()
{
	const float PI = 3.1415f;
	const int DIM = 512;
	const int N = (DIM * DIM);
	const float SPEED = 1.0f; // Should be between 0.0 and 1.0
	const float MAX_TEMP = 1.0f;
	const float MIN_TEMP = 0.000001f;
	const float GRID_TEMP = (MAX_TEMP + MIN_TEMP) / 2;

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
	shader.setInt("heatmap", 0);

	DataBlock data;

	// Create quad
	unsigned int VAO, VBO, EBO;
	createQuad(&VAO, &VBO, &EBO);

	// Create texture in DataBlock
	createTexture(&data.textureID, DIM, DIM);

	// Register OpenGL texture with CUDA
	HANDLE_ERROR(cudaGraphicsGLRegisterImage(&data.resource, data.textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

	// Select CUDA device
	selectDevice(0);

	// Host Alloc
	float* outSrc;
	HANDLE_ERROR(cudaMallocHost((void**)&outSrc, N * sizeof(float)));

	// Default temp for grid
	init_anim(&data, DIM, GRID_TEMP, blocksPerGrid, threadsPerBlock);

	// Heater 1
	Heater heater1;
	heater1.temp = 1.0f;
	heater1.x =		128;
	heater1.y =		256;
	heater1.radius = 50;
	add_heater(&data, &heater1, DIM, blocksPerGrid, threadsPerBlock);

	//Heater 2 (Cooler)
	Heater heater2;
	heater2.temp = 0.0f;
	heater2.x = 384;
	heater2.y = 256;
	heater2.radius = 50;
	add_heater(&data, &heater2, DIM, blocksPerGrid, threadsPerBlock);

	// Main loop
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);
		animate(&data, blocksPerGrid, threadsPerBlock, SPEED, DIM);

		// Print FPS every second, i use the cudaEvent instead of chrono on purpose
		if (data.elapsedTime >= 1000) {
			float fps = data.frames / (data.elapsedTime / 1000.0f);
			printf("Total Time Elapsed: %1.f seconds, FPS: %.1f\n", (data.totalTime/1000), fps);

			// Reset timer and frame count
			data.elapsedTime = 0.0f;
			data.frames = 0;
		}
		
		glBindTexture(GL_TEXTURE_2D, data.textureID);
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Free memory
	anim_exit(&data);
	HANDLE_ERROR(cudaFreeHost(outSrc));
	glfwTerminate();
}