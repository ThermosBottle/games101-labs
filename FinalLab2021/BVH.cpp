#include <algorithm>
#include <cassert>
#include "BVH.hpp"

namespace
{

    // const SAH bucket count for partitioning
    constexpr int kSAHBucketCount = 16;

    struct BucketInfo
    {
        int count = 0;
        Bounds3 bounds;
    };

    int getBucketIndex(const Bounds3 &centroidBounds, const Vector3f &centroid, int dim)
    {
        Vector3f offset = centroidBounds.Offset(centroid);
        double axisOffset = dim == 0 ? offset.x : (dim == 1 ? offset.y : offset.z);
        int bucket = static_cast<int>(kSAHBucketCount * axisOffset);
        if (bucket == kSAHBucketCount)
            bucket = kSAHBucketCount - 1;
        if (bucket < 0)
            bucket = 0;
        return bucket;
    }

    void sortByCentroid(std::vector<Object *> &objects, int dim)
    {
        switch (dim)
        {
        case 0:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                      { return f1->getBounds().Centroid().x < f2->getBounds().Centroid().x; });
            break;
        case 1:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                      { return f1->getBounds().Centroid().y < f2->getBounds().Centroid().y; });
            break;
        default:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                      { return f1->getBounds().Centroid().z < f2->getBounds().Centroid().z; });
            break;
        }
    }

    std::pair<std::vector<Object *>, std::vector<Object *>> splitByMedian(std::vector<Object *> objects, int dim)
    {
        sortByCentroid(objects, dim);

        auto mid = objects.begin() + (objects.size() / 2);
        std::vector<Object *> left(objects.begin(), mid);
        std::vector<Object *> right(mid, objects.end());
        return {left, right};
    }

} // namespace

BVHAccel::BVHAccel(std::vector<Object *> p, int maxPrimsInNode,
                   SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod),
      primitives(std::move(p))
{
    time_t start, stop;
    time(&start);
    if (primitives.empty())
        return;

    root = recursiveBuild(primitives);

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    printf(
        "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
        hrs, mins, secs);
}

BVHBuildNode *BVHAccel::recursiveBuild(std::vector<Object *> objects)
{
    BVHBuildNode *node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    if (objects.size() == 1)
    {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        node->area = objects[0]->getArea();
        return node;
    }
    else if (objects.size() == 2)
    {
        node->left = recursiveBuild(std::vector{objects[0]});
        node->right = recursiveBuild(std::vector{objects[1]});

        node->bounds = Union(node->left->bounds, node->right->bounds);
        node->area = node->left->area + node->right->area;
        return node;
    }
    else
    {
        Bounds3 centroidBounds;
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
                Union(centroidBounds, objects[i]->getBounds().Centroid());
        int dim = centroidBounds.maxExtent();
        std::vector<Object *> leftshapes;
        std::vector<Object *> rightshapes;

        if (splitMethod == SplitMethod::SAH)
        {
            Vector3f diagonal = centroidBounds.Diagonal();
            // to check if we can split the objects into buckets along the chosen dimension
            bool canBucketSplit = (dim == 0 && diagonal.x > 0) ||
                                  (dim == 1 && diagonal.y > 0) ||
                                  (dim == 2 && diagonal.z > 0);

            if (canBucketSplit)
            {
                std::array<BucketInfo, kSAHBucketCount> buckets;
                for (Object *object : objects)
                {
                    Bounds3 objectBounds = object->getBounds();
                    int bucketIndex = getBucketIndex(centroidBounds, objectBounds.Centroid(), dim);
                    buckets[bucketIndex].count++;
                    buckets[bucketIndex].bounds = Union(buckets[bucketIndex].bounds, objectBounds);
                }

                std::array<Bounds3, kSAHBucketCount> prefixBounds;
                std::array<Bounds3, kSAHBucketCount> suffixBounds;
                std::array<int, kSAHBucketCount> prefixCount{};
                std::array<int, kSAHBucketCount> suffixCount{};

                Bounds3 runningPrefixBounds;
                int runningPrefixCount = 0;
                for (int i = 0; i < kSAHBucketCount; ++i)
                {
                    runningPrefixCount += buckets[i].count;
                    runningPrefixBounds = Union(runningPrefixBounds, buckets[i].bounds);
                    prefixCount[i] = runningPrefixCount;
                    prefixBounds[i] = runningPrefixBounds;
                }

                Bounds3 runningSuffixBounds;
                int runningSuffixCount = 0;
                for (int i = kSAHBucketCount - 1; i >= 0; --i)
                {
                    runningSuffixCount += buckets[i].count;
                    runningSuffixBounds = Union(runningSuffixBounds, buckets[i].bounds);
                    suffixCount[i] = runningSuffixCount;
                    suffixBounds[i] = runningSuffixBounds;
                }

                double parentArea = bounds.SurfaceArea();
                double minCost = std::numeric_limits<double>::infinity();
                int bestSplitBucket = -1;

                if (parentArea > 0)
                {
                    for (int i = 0; i < kSAHBucketCount - 1; ++i)
                    {
                        if (prefixCount[i] == 0 || suffixCount[i + 1] == 0)
                            continue;

                        double leftArea = prefixBounds[i].SurfaceArea();
                        double rightArea = suffixBounds[i + 1].SurfaceArea();
                        double cost = 1.0 +
                                      (leftArea * prefixCount[i] + rightArea * suffixCount[i + 1]) /
                                          parentArea;

                        if (cost < minCost)
                        {
                            minCost = cost;
                            bestSplitBucket = i;
                        }
                    }
                }

                if (bestSplitBucket != -1)
                {
                    leftshapes.reserve(objects.size());
                    rightshapes.reserve(objects.size());
                    for (Object *object : objects)
                    {
                        int bucketIndex = getBucketIndex(centroidBounds, object->getBounds().Centroid(), dim);
                        if (bucketIndex <= bestSplitBucket)
                            leftshapes.push_back(object);
                        else
                            rightshapes.push_back(object);
                    }
                }
            }
        }

        if (leftshapes.empty() || rightshapes.empty())
        {
            auto medianSplit = splitByMedian(objects, dim);
            leftshapes = std::move(medianSplit.first);
            rightshapes = std::move(medianSplit.second);
        }

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = recursiveBuild(leftshapes);
        node->right = recursiveBuild(rightshapes);

        node->bounds = Union(node->left->bounds, node->right->bounds);
    }

    return node;
}

Intersection BVHAccel::Intersect(const Ray &ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

Intersection BVHAccel::getIntersection(BVHBuildNode *node, const Ray &ray) const
{
    // TODO Traverse the BVH to find intersection
    std::array<int, 3> dirIsNeg = {
        int(ray.direction.x < 0),
        int(ray.direction.y < 0),
        int(ray.direction.z < 0)};
    if (!node->bounds.IntersectP(ray, ray.direction_inv, dirIsNeg))
        return Intersection();
    if (node->left == nullptr && node->right == nullptr)
        return node->object->getIntersection(ray);
    Intersection leftIsect = node->left ? getIntersection(node->left, ray) : Intersection();
    Intersection rightIsect = node->right ? getIntersection(node->right, ray) : Intersection();
    if (leftIsect.happened && rightIsect.happened)
        // return the closer intersection
        return leftIsect.distance < rightIsect.distance ? leftIsect : rightIsect;
    else if (leftIsect.happened)
        return leftIsect;
    else
        return rightIsect;
}

void BVHAccel::getSample(BVHBuildNode *node, float p, Intersection &pos, float &pdf)
{
    if (node->left == nullptr || node->right == nullptr)
    {
        node->object->Sample(pos, pdf);
        pdf *= node->area;
        return;
    }
    if (p < node->left->area)
        getSample(node->left, p, pos, pdf);
    else
        getSample(node->right, p - node->left->area, pos, pdf);
}

void BVHAccel::Sample(Intersection &pos, float &pdf)
{
    // float p = std::sqrt(get_random_float()) * root->area;
    float p = get_random_float() * root->area;
    // 1-dimension not need to sqrt
    getSample(root, p, pos, pdf);
    pdf /= root->area;
}