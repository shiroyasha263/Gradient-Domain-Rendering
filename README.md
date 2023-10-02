Gradient Domain Renderer
===============================

Physically based gradient domain renderer with various other algorithms implemented alongside it. This is built over PBRT v-4


Features
--------
1. Gradient Domain Rendering, with path reconnection
   - ![Gradient Domain Rendering - 128 spp - Cornell Box](https://github.com/shiroyasha263/Gradient-Domain-Rendering/blob/GDR/Results/GDR-128-pic.png)
   - ![Gradient Domain Rendering - 128 spp - Teapot](https://github.com/shiroyasha263/Gradient-Domain-Rendering/assets/63004181/62a7d86d-b4dc-4757-850f-322a251f0a1c)

2. Stochastic Lightcuts, with VPL generation for indirect lighting
   - ![Stochastic Lightcuts - 128 spp - Cornell Box - 620s](https://github.com/shiroyasha263/Gradient-Domain-Rendering/assets/63004181/1af8cf79-c5e1-4578-aa14-30b5032f903b)
   - ![Stochastic Lightcuts - 128 spp - Staircase - 1000s](https://github.com/shiroyasha263/Gradient-Domain-Rendering/assets/63004181/39e53d76-4daa-4ed3-ad5c-99aed39d8651)

3. Gradient Domain Rendering with Stochastic Lightcuts Shiftmapping
   - ![GDRSLC - 64 spp - Cornell Box - 620s](https://github.com/shiroyasha263/Gradient-Domain-Rendering/assets/63004181/9abeb9e3-0cf4-4b97-a3f0-ba987909216c)
   - ![GDRSLC - 64 spp - Staircase - 820s](https://github.com/shiroyasha263/Gradient-Domain-Rendering/assets/63004181/98bbd15c-e07a-4075-8e29-a2a56da099f9)



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

Bug Reports and PRs
-------------------

Please use the [pbrt-v4 github issue
tracker](https://github.com/mmp/pbrt-v4/issues) to report bugs in pbrt-v4.
(We have pre-populated it with a number of issues corresponding to known
bugs in the initial release.)

We are always happy to receive pull requests that fix bugs, including bugs
you find yourself or fixes for open issues in the issue tracker.  We are
also happy to hear suggestions about improvements to the implementations of
the various algorithms we have implemented.

Note, however, that in the interests of finishing the book in a finite
amount of time, the functionality of pbrt-v4 is basically fixed at this
point.  We therefore will not be accepting PRs that make major changes to the
system's operation or structure (but feel free to keep them in your own
forks!).  Also, don't bother sending PRs for anything marked "TODO" or
"FIXME" in the source code; we'll take care of those as we finish polishing
things up.

Updating pbrt-v3 scenes
-----------------------

There are a variety of changes to the input file format and, as noted
above, the new format is not yet documented.  However, pbrt-v4 partially
makes up for that by providing an automatic upgrade mechanism:
```bash
$ pbrt --upgrade old.pbrt > new.pbrt
```

Most scene files can be automatically updated. In some cases manual
intervention is required; an error message will be printed in this case.

The environment map parameterization has also changed (from equi-rect to an
equi-area mapping); you can upgrade environment maps using
```bash
$ imgtool makeequiarea old.exr --outfile new.exr
```

Converting scenes to pbrt's file format
---------------------------------------

The best option for importing scenes to pbrt is to use
[assimp](https://www.assimp.org/), which as of January 21, 2021 includes
support for exporting to pbrt-v4's file format:
```bash
$ assimp export scene.fbx scene.pbrt
```

While the converter tries to convert materials to pbrt's material model,
some manual tweaking may be necessary after export.  Furthermore, area
light sources are not always successfully detected; manual intervention may
be required for them as well.  Use of pbrt's built-in support for
converting meshes to use the binary PLY format is also recommended after
conversion. (`pbrt --toply scene.pbrt > newscene.pbrt`).
