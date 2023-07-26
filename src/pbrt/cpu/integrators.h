// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_INTEGRATORS_H
#define PBRT_CPU_INTEGRATORS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/camera.h>
#include <pbrt/base/sampler.h>
#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/cpu/primitive.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace pbrt {

struct PrimalRay {
    PrimalRay(RayDifferential r) {
        specularBounce = true;
        reconPossible = false;
        live = true;
        beta = SampledSpectrum(1.f);
        ray = r;
        depth = 0;
        pathL = {};
        bs = true;
        noHit = false;
        prevSpecular = true;
    }
    bool specularBounce;
    bool reconPossible;
    bool live;
    bool bs;
    bool noHit;
    bool prevSpecular;
    RayDifferential ray;
    SampledSpectrum beta, Lin, prevMul;
    Normal3f prevN;
    Float prevCosine;
    Float prevD, pdf, prevPDF;
    int depth;
    std::vector<SampledSpectrum> pathL;
};

struct ShiftRay {
    ShiftRay(RayDifferential r) {
        specularBounce = true;
        live = true;
        reconnected = false;
        beta = SampledSpectrum(1.f);
        ray = r;
        depth = 0;
        prevMul = SampledSpectrum(1.f);
        pathL = {};
    }
    bool specularBounce;
    bool live;
    bool reconnected;
    RayDifferential ray;
    SampledSpectrum beta, prevMul;
    Normal3f prevN;
    BSDF prevBSDF;
    Vector3f prevW;
    int depth;
    std::vector<SampledSpectrum> pathL;
    std::vector<Float> weight;
    Float reconMIS;
};

struct VPL {
    VPL() { 
        I = SampledSpectrum(0.f);
        isect = {};
        ray = {};
        lambda = SampledWavelengths();
        point = Point3f(0.f, 0.f, 0.f);
    }
    VPL(SampledSpectrum I, SurfaceInteraction isect, RayDifferential ray, SampledWavelengths lambda, Point3f point)
        : I(I), isect(isect), ray(ray), lambda(lambda), point(point) {}
    SampledSpectrum I;
    RayDifferential ray;
    SampledWavelengths lambda;
    SurfaceInteraction isect;
    Point3f point;
};

struct VPLTreeNodes {
    VPLTreeNodes(){
        vpl = NULL;
        left = NULL;
        right = NULL;
        I = SampledSpectrum(0.f);
        prob = 1.f;
        boundMax = Point3f(0.f, 0.f, 0.f);
        boundMin = Point3f(0.f, 0.f, 0.f);
    }
    VPLTreeNodes(VPL *vpl, VPLTreeNodes *left, VPLTreeNodes *right, SampledSpectrum I, VPLTreeNodes *parent = NULL)
    :vpl(vpl), left(left), right(right), I(I){
        prob = 0.f;
        if (vpl != NULL) {
            boundMin = vpl->point;
            boundMax = vpl->point;
        } else {
            boundMax = Point3f(0.f, 0.f, 0.f);
            boundMin = Point3f(0.f, 0.f, 0.f);
        }
    }
    VPL *vpl;
    VPLTreeNodes *left;
    VPLTreeNodes *right;
    SampledSpectrum I;
    Point3f boundMin, boundMax;
    bool sampledLeft = true;
    unsigned int depth = 0;
    float prob = 1.f;
};

struct CutNodes {
    CutNodes(){
        TreeNode = NULL;
        vpl = NULL;
    }
    CutNodes(VPLTreeNodes *TreeNode = NULL, VPL *vpl = NULL, Float error = 0.f, Float prob = 1.f)
    :TreeNode(TreeNode), vpl(vpl), error(prob), prob(error) {}
    VPLTreeNodes *TreeNode;
    VPL *vpl;
    Float error = 0.f;
    Float prob = 1.f;
    Float probP = 1.f;
    Float jacobianP = 1.f;
    Float jacobianS = 1.f;
};

// Integrator Definition
class Integrator {
  public:
    // Integrator Public Methods
    virtual ~Integrator();

    static std::unique_ptr<Integrator> Create(
        const std::string &name, const ParameterDictionary &parameters, Camera camera,
        Sampler sampler, Primitive aggregate, std::vector<Light> lights,
        const RGBColorSpace *colorSpace, const FileLoc *loc);

    virtual std::string ToString() const = 0;

    virtual void Render() = 0;

    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const;
    bool IntersectP(const Ray &ray, Float tMax = Infinity) const;

    bool Unoccluded(const Interaction &p0, const Interaction &p1) const {
        return !IntersectP(p0.SpawnRayTo(p1), 1 - ShadowEpsilon);
    }

    SampledSpectrum Tr(const Interaction &p0, const Interaction &p1,
                       const SampledWavelengths &lambda) const;

    // Integrator Public Members
    Primitive aggregate;
    std::vector<Light> lights;
    std::vector<Light> infiniteLights;

  protected:
    // Integrator Protected Methods
    Integrator(Primitive aggregate, std::vector<Light> lights)
        : aggregate(aggregate), lights(lights) {
        // Integrator constructor implementation
        Bounds3f sceneBounds = aggregate ? aggregate.Bounds() : Bounds3f();
        LOG_VERBOSE("Scene bounds %s", sceneBounds);
        for (auto &light : lights) {
            light.Preprocess(sceneBounds);
            if (light.Type() == LightType::Infinite)
                infiniteLights.push_back(light);
        }
    }
};

// ImageTileIntegrator Definition
class ImageTileIntegrator : public Integrator {
  public:
    // ImageTileIntegrator Public Methods
    ImageTileIntegrator(Camera camera, Sampler sampler, Primitive aggregate,
                        std::vector<Light> lights)
        : Integrator(aggregate, lights), camera(camera), samplerPrototype(sampler) {}

    void Render();

    virtual void EvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                                     ScratchBuffer &scratchBuffer) = 0;

  protected:
    // ImageTileIntegrator Protected Members
    Camera camera;
    Sampler samplerPrototype;
};

// RayIntegrator Definition
class RayIntegrator : public ImageTileIntegrator {
  public:
    // RayIntegrator Public Methods
    RayIntegrator(Camera camera, Sampler sampler, Primitive aggregate,
                  std::vector<Light> lights)
        : ImageTileIntegrator(camera, sampler, aggregate, lights) {}

    void EvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                             ScratchBuffer &scratchBuffer) final;

    virtual SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda,
                               Sampler sampler, ScratchBuffer &scratchBuffer,
                               VisibleSurface *visibleSurface) const = 0;
};

// RandomWalkIntegrator Definition
class RandomWalkIntegrator : public RayIntegrator {
  public:
    // RandomWalkIntegrator Public Methods
    RandomWalkIntegrator(int maxDepth, Camera camera, Sampler sampler,
                         Primitive aggregate, std::vector<Light> lights)
        : RayIntegrator(camera, sampler, aggregate, lights), maxDepth(maxDepth) {}

    static std::unique_ptr<RandomWalkIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const {
        return LiRandomWalk(ray, lambda, sampler, scratchBuffer, 0);
    }

  private:
    // RandomWalkIntegrator Private Methods
    SampledSpectrum LiRandomWalk(RayDifferential ray, SampledWavelengths &lambda,
                                 Sampler sampler, ScratchBuffer &scratchBuffer,
                                 int depth) const {
        // Intersect ray with scene and return if no intersection
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        if (!si) {
            // Return emitted light from infinite light sources
            SampledSpectrum Le(0.f);
            for (Light light : infiniteLights)
                Le += light.Le(ray, lambda);
            return Le;
        }
        SurfaceInteraction &isect = si->intr;

        // Get emitted radiance at surface intersection
        Vector3f wo = -ray.d;
        SampledSpectrum Le = isect.Le(wo, lambda);

        // Terminate random walk if maximum depth has been reached
        if (depth == maxDepth)
            return Le;

        // Compute BSDF at random walk intersection point
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf)
            return Le;

        // Randomly sample direction leaving surface for random walk
        Point2f u = sampler.Get2D();
        Vector3f wp = SampleUniformSphere(u);

        // Evaluate BSDF at surface for sampled direction
        SampledSpectrum fcos = bsdf.f(wo, wp) * AbsDot(wp, isect.shading.n);
        if (!fcos)
            return Le;

        // Recursively trace ray to estimate incident radiance at surface
        ray = isect.SpawnRay(wp);
        return Le + fcos * LiRandomWalk(ray, lambda, sampler, scratchBuffer, depth + 1) /
                        (1 / (4 * Pi));
    }

    // RandomWalkIntegrator Private Members
    int maxDepth;
};

// SimplePathIntegrator Definition
class SimplePathIntegrator : public RayIntegrator {
  public:
    // SimplePathIntegrator Public Methods
    SimplePathIntegrator(int maxDepth, bool sampleLights, bool sampleBSDF, Camera camera,
                         Sampler sampler, Primitive aggregate, std::vector<Light> lights);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<SimplePathIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // SimplePathIntegrator Private Members
    int maxDepth;
    bool sampleLights, sampleBSDF;
    UniformLightSampler lightSampler;
};

// PathIntegrator Definition
class PathIntegrator : public RayIntegrator {
  public:
    // PathIntegrator Public Methods
    PathIntegrator(int maxDepth, Camera camera, Sampler sampler, Primitive aggregate,
                   std::vector<Light> lights,
                   const std::string &lightSampleStrategy = "bvh",
                   bool regularize = false);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<PathIntegrator> Create(const ParameterDictionary &parameters,
                                                  Camera camera, Sampler sampler,
                                                  Primitive aggregate,
                                                  std::vector<Light> lights,
                                                  const FileLoc *loc);

    std::string ToString() const;

  private:
    // PathIntegrator Private Methods
    SampledSpectrum SampleLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, Sampler sampler) const;

    // PathIntegrator Private Members
    int maxDepth;
    LightSampler lightSampler;
    bool regularize;
};

// SimpleVolPathIntegrator Definition
class SimpleVolPathIntegrator : public RayIntegrator {
  public:
    // SimpleVolPathIntegrator Public Methods
    SimpleVolPathIntegrator(int maxDepth, Camera camera, Sampler sampler,
                            Primitive aggregate, std::vector<Light> lights);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<SimpleVolPathIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // SimpleVolPathIntegrator Private Members
    int maxDepth;
};

// VolPathIntegrator Definition
class VolPathIntegrator : public RayIntegrator {
  public:
    // VolPathIntegrator Public Methods
    VolPathIntegrator(int maxDepth, Camera camera, Sampler sampler, Primitive aggregate,
                      std::vector<Light> lights,
                      const std::string &lightSampleStrategy = "bvh",
                      bool regularize = false)
        : RayIntegrator(camera, sampler, aggregate, lights),
          maxDepth(maxDepth),
          lightSampler(LightSampler::Create(lightSampleStrategy, lights, Allocator())),
          regularize(regularize) {}

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<VolPathIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // VolPathIntegrator Private Methods
    SampledSpectrum SampleLd(const Interaction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, Sampler sampler,
                             SampledSpectrum beta, SampledSpectrum inv_w_u) const;

    // VolPathIntegrator Private Members
    int maxDepth;
    LightSampler lightSampler;
    bool regularize;
};

// AOIntegrator Definition
class AOIntegrator : public RayIntegrator {
  public:
    // AOIntegrator Public Methods
    AOIntegrator(bool cosSample, Float maxDist, Camera camera, Sampler sampler,
                 Primitive aggregate, std::vector<Light> lights, Spectrum illuminant);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<AOIntegrator> Create(const ParameterDictionary &parameters,
                                                Spectrum illuminant, Camera camera,
                                                Sampler sampler, Primitive aggregate,
                                                std::vector<Light> lights,
                                                const FileLoc *loc);

    std::string ToString() const;

  private:
    bool cosSample;
    Float maxDist;
    Spectrum illuminant;
    Float illumScale;
};

// LightPathIntegrator Definition
class LightPathIntegrator : public ImageTileIntegrator {
  public:
    // LightPathIntegrator Public Methods
    LightPathIntegrator(int maxDepth, Camera camera, Sampler sampler, Primitive aggregate,
                        std::vector<Light> lights);

    void EvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                             ScratchBuffer &scratchBuffer);

    static std::unique_ptr<LightPathIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // LightPathIntegrator Private Members
    int maxDepth;
    PowerLightSampler lightSampler;
};

// BDPTIntegrator Definition
struct Vertex;
class BDPTIntegrator : public RayIntegrator {
  public:
    // BDPTIntegrator Public Methods
    BDPTIntegrator(Camera camera, Sampler sampler, Primitive aggregate,
                   std::vector<Light> lights, int maxDepth, bool visualizeStrategies,
                   bool visualizeWeights, bool regularize = false)
        : RayIntegrator(camera, sampler, aggregate, lights),
          maxDepth(maxDepth),
          regularize(regularize),
          lightSampler(new PowerLightSampler(lights, Allocator())),
          visualizeStrategies(visualizeStrategies),
          visualizeWeights(visualizeWeights) {}

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;

    static std::unique_ptr<BDPTIntegrator> Create(const ParameterDictionary &parameters,
                                                  Camera camera, Sampler sampler,
                                                  Primitive aggregate,
                                                  std::vector<Light> lights,
                                                  const FileLoc *loc);

    std::string ToString() const;

    void Render();

  private:
    // BDPTIntegrator Private Members
    int maxDepth;
    bool regularize;
    LightSampler lightSampler;
    bool visualizeStrategies, visualizeWeights;
    mutable std::vector<Film> weightFilms;
};

// MLTIntegrator Definition
class MLTSampler;

class MLTIntegrator : public Integrator {
  public:
    // MLTIntegrator Public Methods
    MLTIntegrator(Camera camera, Primitive aggregate, std::vector<Light> lights,
                  int maxDepth, int nBootstrap, int nChains, int mutationsPerPixel,
                  Float sigma, Float largeStepProbability, bool regularize)
        : Integrator(aggregate, lights),
          lightSampler(new PowerLightSampler(lights, Allocator())),
          camera(camera),
          maxDepth(maxDepth),
          nBootstrap(nBootstrap),
          nChains(nChains),
          mutationsPerPixel(mutationsPerPixel),
          sigma(sigma),
          largeStepProbability(largeStepProbability),
          regularize(regularize) {}

    void Render();

    static std::unique_ptr<MLTIntegrator> Create(const ParameterDictionary &parameters,
                                                 Camera camera, Primitive aggregate,
                                                 std::vector<Light> lights,
                                                 const FileLoc *loc);

    std::string ToString() const;

  private:
    // MLTIntegrator Constants
    static constexpr int cameraStreamIndex = 0;
    static constexpr int lightStreamIndex = 1;
    static constexpr int connectionStreamIndex = 2;
    static constexpr int nSampleStreams = 3;

    // MLTIntegrator Private Methods
    SampledSpectrum L(ScratchBuffer &scratchBuffer, MLTSampler &sampler, int k,
                      Point2f *pRaster, SampledWavelengths *lambda);

    static Float c(const SampledSpectrum &L, const SampledWavelengths &lambda) {
        return L.y(lambda);
    }

    // MLTIntegrator Private Members
    Camera camera;
    bool regularize;
    LightSampler lightSampler;
    int maxDepth, nBootstrap;
    int mutationsPerPixel;
    Float sigma, largeStepProbability;
    int nChains;
};

// SPPMIntegrator Definition
class SPPMIntegrator : public Integrator {
  public:
    // SPPMIntegrator Public Methods
    SPPMIntegrator(Camera camera, Sampler sampler, Primitive aggregate,
                   std::vector<Light> lights, int photonsPerIteration, int maxDepth,
                   Float initialSearchRadius, int seed, const RGBColorSpace *colorSpace)
        : Integrator(aggregate, lights),
          camera(camera),
          samplerPrototype(sampler),
          initialSearchRadius(initialSearchRadius),
          maxDepth(maxDepth),
          photonsPerIteration(photonsPerIteration > 0
                                  ? photonsPerIteration
                                  : camera.GetFilm().PixelBounds().Area()),
          colorSpace(colorSpace),
          digitPermutationsSeed(seed) {}

    static std::unique_ptr<SPPMIntegrator> Create(const ParameterDictionary &parameters,
                                                  const RGBColorSpace *colorSpace,
                                                  Camera camera, Sampler sampler,
                                                  Primitive aggregate,
                                                  std::vector<Light> lights,
                                                  const FileLoc *loc);

    std::string ToString() const;

    void Render();

  private:
    // SPPMIntegrator Private Methods
    SampledSpectrum SampleLd(const SurfaceInteraction &intr, const BSDF &bsdf,
                             SampledWavelengths &lambda, Sampler sampler,
                             LightSampler lightSampler) const;

    // SPPMIntegrator Private Members
    Camera camera;
    Float initialSearchRadius;
    Sampler samplerPrototype;
    int digitPermutationsSeed;
    int maxDepth;
    int photonsPerIteration;
    const RGBColorSpace *colorSpace;
};

class GradientIntegrator : public Integrator {
  public:
    GradientIntegrator(Camera camera, Sampler sampler, Primitive aggregate,
        std::vector<Light> lights, int maxDepth)
        : Integrator(aggregate, lights),
          camera(camera),
          samplerPrototype(sampler), maxDepth(maxDepth), lightSampler(lights, Allocator()) {
        Primal = std::vector<std::vector<SampledSpectrum>>(
            camera.GetFilm().FullResolution().x,
            std::vector<SampledSpectrum>(camera.GetFilm().FullResolution().y,
                                         SampledSpectrum(0.f)));
        Temp = std::vector<std::vector<SampledSpectrum>>(
            camera.GetFilm().FullResolution().x,
            std::vector<SampledSpectrum>(camera.GetFilm().FullResolution().y,
                                         SampledSpectrum(0.f)));
        xGrad = std::vector<std::vector<SampledSpectrum>>(
            camera.GetFilm().FullResolution().x + 1,
            std::vector<SampledSpectrum>(camera.GetFilm().FullResolution().y,
                                         SampledSpectrum(0.f)));
        yGrad = std::vector<std::vector<SampledSpectrum>>(
            camera.GetFilm().FullResolution().x,
            std::vector<SampledSpectrum>(camera.GetFilm().FullResolution().y + 1,
                                         SampledSpectrum(0.f)));
    }

    static std::unique_ptr<GradientIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

    void Render();

    void EvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                             ScratchBuffer &scratchBuffer);

    void GradEvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                             ScratchBuffer &scratchBuffer);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface) const;
    void PrimalRayPropogate(PrimalRay &pRay, SampledWavelengths &lambda, Sampler sampler,
                            ScratchBuffer &scratchBuffer, VisibleSurface *,
                            Float randomStorage[]) const;
    void ShiftRayPropogate(ShiftRay &sRay, SampledWavelengths &lambda, Sampler sampler,
                            ScratchBuffer &scratchBuffer, VisibleSurface *,
                           Float randomStorage[], const PrimalRay &pRay) const;
  private:
    Camera camera;
    Sampler samplerPrototype;
    int maxDepth;
    UniformLightSampler lightSampler;
    std::vector<std::vector<SampledSpectrum>> Primal, xGrad, yGrad, Temp;
};

class VPLIntegrator : public Integrator {
  public:
    VPLIntegrator(int maxDepth, Camera camera, Sampler sampler, Primitive aggregate,
                  std::vector<Light> lights,
                  const std::string &lightSampleStrategy = "bvh",
                  bool regularize = false);

    static std::unique_ptr<VPLIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

    void Render();

    void EvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                             ScratchBuffer &scratchBuffer);

    void PixelSampleVPLGenerator(int maxVPL, Sampler sampler, ScratchBuffer &scratchBuffer);

    void VPLTreeGenerator();

    VPLTreeNodes SampleTree(VPLTreeNodes *vplNode, const SurfaceInteraction &intr,
                            const BSDF *bsdf, Float randomNumber,
                            ScratchBuffer &scratchBuffer, float &prob);

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, Sampler sampler,
                       ScratchBuffer &scratchBuffer,
                       VisibleSurface *visibleSurface);

    Float NodeError(Point3f shadingPoint, Normal3f shadingNormal, Point3f BBMin,
                    Point3f BBMax);

    void GenerateCuts(int cutSize, const SurfaceInteraction &intr,
                 std::vector<CutNodes> &cutNodes);

    void SampleCuts(int cutSize, const SurfaceInteraction &intr, const BSDF *bsdf,
                    Sampler sampler, ScratchBuffer &scratchBuffer,
                    std::vector<CutNodes> &cutNodes);

  private:
    // PathIntegrator Private Methods
    SampledSpectrum SampleLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, Sampler sampler) const;
    SampledSpectrum SampleVPLLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                             SampledWavelengths &lambda, Sampler sampler, ScratchBuffer &scratchBuffer);
    Float MinimumDistance(Point3f sample, Point3f BBMin, Point3f BBMax);
    Float MaximumCosine(Point3f shadingPoint, Normal3f shadingNormal, Point3f BBMin,
                        Point3f BBMax);
    Camera camera;
    Sampler samplerPrototype;
    int maxDepth;
    LightSampler lightSampler;
    bool regularize;
    std::vector<VPL> VPLList;
    std::vector<std::vector<VPLTreeNodes>> VPLTree;
};

class VPLGradient : public Integrator {
  public:
    VPLGradient(Camera camera, Sampler sampler, Primitive aggregate,
                       std::vector<Light> lights, int maxDepth)
        : Integrator(aggregate, lights),
          camera(camera),
          samplerPrototype(sampler),
          maxDepth(maxDepth),
          lightSampler(lights, Allocator()) {
        Primal = std::vector<std::vector<SampledSpectrum>>(
            camera.GetFilm().FullResolution().x,
            std::vector<SampledSpectrum>(camera.GetFilm().FullResolution().y,
                                         SampledSpectrum(0.f)));
        Temp = std::vector<std::vector<SampledSpectrum>>(
            camera.GetFilm().FullResolution().x,
            std::vector<SampledSpectrum>(camera.GetFilm().FullResolution().y,
                                         SampledSpectrum(0.f)));
        xGrad = std::vector<std::vector<SampledSpectrum>>(
            camera.GetFilm().FullResolution().x + 1,
            std::vector<SampledSpectrum>(camera.GetFilm().FullResolution().y,
                                         SampledSpectrum(0.f)));
        yGrad = std::vector<std::vector<SampledSpectrum>>(
            camera.GetFilm().FullResolution().x,
            std::vector<SampledSpectrum>(camera.GetFilm().FullResolution().y + 1,
                                         SampledSpectrum(0.f)));

        VPLList = {};
        VPLTree = {};
    }

    static std::unique_ptr<VPLGradient> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const;

    void Render();

    void GradEvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                                 ScratchBuffer &scratchBuffer);

    void PixelSampleVPLGenerator(int maxVPL, Sampler sampler,
                                 ScratchBuffer &scratchBuffer, std::vector<VPL> &vplList);

    void VPLTreeGenerator();

    VPLTreeNodes SampleTree(VPLTreeNodes *vplNode, const SurfaceInteraction &intr,
                            const BSDF *bsdf, Float randomNumber,
                            ScratchBuffer &scratchBuffer, float &prob);

    void SampleTreeCuts(int cutSize, const SurfaceInteraction &intr, const BSDF *bsdf,
                        Sampler sampler, ScratchBuffer &scratchBuffer,
                        std::vector<VPLTreeNodes> &samplePoints);

    void GenerateCuts(int cutSize, const SurfaceInteraction &intr,
                      std::vector<CutNodes> &cutNodes);

    void SampleCuts(int cutSize, const SurfaceInteraction &intr, const BSDF *bsdf,
                    Sampler sampler, ScratchBuffer &scratchBuffer,
                    std::vector<CutNodes> &cutNodes);

    void SampleShiftedCuts(int cutSize, const SurfaceInteraction &intr, const BSDF *bsdf,
                           Sampler sampler, ScratchBuffer &scratchBuffer,
                           std::vector<CutNodes> &cutNodes);

    void PrimalRayPropogate(PrimalRay &pRay, SampledWavelengths &lambda, Sampler sampler,
                            ScratchBuffer &scratchBuffer, VisibleSurface *,
                            std::vector<CutNodes> &cutNodes);
    void ShiftRayPropogate(ShiftRay &sRay, SampledWavelengths &lambda, Sampler sampler,
                           ScratchBuffer &scratchBuffer, VisibleSurface *,
                           std::vector<CutNodes> &cutNodes, const PrimalRay &pRay);

    void SampleVPLLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                                SampledWavelengths &lambda, Sampler sampler,
                                ScratchBuffer &scratchBuffer,
                                std::vector<CutNodes> &cutNodes, PrimalRay &pRay);

    void SampleVPLLdShifted(const SurfaceInteraction &intr, const BSDF *bsdf,
                                       SampledWavelengths &lambda, Sampler sampler,
                                       ScratchBuffer &scratchBuffer,
                                       std::vector<CutNodes> &cutNodes, ShiftRay &sRay);

    Float MinimumDistance(Point3f sample, Point3f BBMin, Point3f BBMax);

    Float MaximumCosine(Point3f shadingPoint, Normal3f shadingNormal, Point3f BBMin,
                        Point3f BBMax);

    Float NodeError(Point3f shadingPoint, Normal3f shadingNormal, Point3f BBMin,
                    Point3f BBMax);

  private:
    Camera camera;
    Sampler samplerPrototype;
    int maxDepth;
    UniformLightSampler lightSampler;
    std::vector<std::vector<SampledSpectrum>> Primal, xGrad, yGrad, Temp;
    std::vector<VPL> VPLList;
    std::vector<std::vector<VPLTreeNodes>> VPLTree;
};

// FunctionIntegrator Definition
class FunctionIntegrator : public Integrator {
  public:
    FunctionIntegrator(std::function<double(Point2f)> func,
                       const std::string &outputFilename, Camera camera, Sampler sampler,
                       bool skipBad, std::string imageFilename);

    static std::unique_ptr<FunctionIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        const FileLoc *loc);

    void Render();

    std::string ToString() const;

  private:
    std::function<double(Point2f)> func;
    std::string outputFilename;
    Camera camera;
    Sampler baseSampler;
    bool skipBad;
    std::string imageFilename;
};

}  // namespace pbrt

#endif  // PBRT_CPU_INTEGRATORS_H
