-- Include the premake5 CUDA module
require('premake5-cuda')

workspace "CUDA"
    architecture "x64"
    startproject "CUDA"

    configurations
	{
		"Debug",
		"Release"
	}
	
	flags
	{
		"MultiProcessorCompile"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

include "CUDA"