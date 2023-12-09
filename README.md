Gradient Domain Renderer
===============================

Physically based gradient domain renderer with various other algorithms implemented alongside it. This is built over PBRT v-4


Features
--------
1. Gradient Domain Rendering, with path reconnection
   - ![Gradient Domain Rendering - 128 spp - Cornell Box](https://github.com/shiroyasha263/Gradient-Domain-Rendering/blob/GDR/Results/GDR-128-pic.png)
   - ![Gradient Domain Rendering - 128 spp - Teapot](https://github.com/shiroyasha263/Gradient-Domain-Rendering/blob/GDR/Results/Grad-Teapot-128-pic.png)

2. Stochastic Lightcuts, with VPL generation for indirect lighting
   - ![Stochastic Lightcuts - 128 spp - Cornell Box - 620s](https://github.com/shiroyasha263/Gradient-Domain-Rendering/blob/GDR/Results/Stochastic%20Lightcuts%20-%20128%20spp%20-%20Cornell%20Box%20-%20620s.png)
   - ![Stochastic Lightcuts - 128 spp - Staircase - 1000s](https://github.com/shiroyasha263/Gradient-Domain-Rendering/blob/GDR/Results/Stochastic%20Lightcuts%20-%20128%20spp%20-%20Staircase%20-%201000s.png)

3. Gradient Domain Rendering with Stochastic Lightcuts Shiftmapping
   - ![GDRSLC - 64 spp - Cornell Box - 620s](https://github.com/shiroyasha263/Gradient-Domain-Rendering/blob/GDR/Results/GDRSLC%20-%2064%20spp%20-%20Cornell%20Box%20-%20620s.png)
   - ![GDRSLC - 64 spp - Staircase - 820s](https://github.com/shiroyasha263/Gradient-Domain-Rendering/blob/GDR/Results/GDRSLC%20-%2064%20spp%20-%20Staircase%20-%20820s.png)


Building the code
-----------------

As before, pbrt uses git submodules for a number of third-party libraries
that it depends on.  Therefore, be sure to use the `--recursive` flag when
cloning the repository:
```bash
$ git clone --recursive https://github.com/mmp/pbrt-v4.git
```

If you accidentally clone pbrt without using ``--recursive`` (or to update
the pbrt source tree after a new submodule has been added, run the
following command to also fetch the dependencies:
```bash
$ git submodule update --init --recursive
```

pbrt uses [cmake](http://www.cmake.org/) for its build system.  Note that a
release build is the default; provide `-DCMAKE_BUILD_TYPE=Debug` to cmake
for a debug build.

pbrt should build on any system that has C++ compiler with support for
C++17; we have verified that it builds on Ubuntu 20.04, MacOS 10.14, and
Windows 10.  We welcome PRs that fix any issues that prevent it from
building on other systems.
