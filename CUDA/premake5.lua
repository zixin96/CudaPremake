project "CUDA"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++11"

	targetdir ("../bin/" .. outputdir .. "/%{prj.name}")
	objdir ("../bin-int/" .. outputdir .. "/%{prj.name}")

	files
	{
		"src/**.h",
		"src/**.cpp",
		"common/**.h",
		"common/**.cpp"
	}

	libdirs
	{
		"vendor/libs"
	}

	includedirs
	{
		"src",
		"vendor/includes"
	}

	links { "glut64.lib" }

    -- Add necessary build customization using standard Premake5
    -- This assumes you have installed Visual Studio integration for CUDA
    -- Here we have it set to 11.6
    buildcustomizations "BuildCustomizations/CUDA 11.6"
	
    -- CUDA specific properties
    cudaFiles 
	{
		"CUDA/src/**.cu"
	} -- files NVCC compiles (Note path starts from the root)
    cudaMaxRegCount "32"

    -- Let's compile for all supported architectures (and also in parallel with -t0)
    cudaCompilerOptions {"-arch=sm_52", "-gencode=arch=compute_52,code=sm_52", "-gencode=arch=compute_60,code=sm_60",
    "-gencode=arch=compute_61,code=sm_61", "-gencode=arch=compute_70,code=sm_70",
    "-gencode=arch=compute_75,code=sm_75", "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_86,code=compute_86", "-t0"}       

	filter "system:windows"
		systemversion "latest"

		-- copy dlls to CUDA binary folder
		postbuildcommands
		{
			("{COPY} dlls/ ../bin/" .. outputdir .. "/CUDA")
		}

	filter "configurations:Debug"
		runtime "Debug"
		symbols "on"
		optimize "off"

	filter "configurations:Release"
		runtime "Release"
        cudaFastMath "On" -- enable fast math for release
        optimize "on"