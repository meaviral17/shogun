/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein,
 *          Viktor Gal, Evan Shelhamer
 */

#include <shogun/machine/DistanceMachine.h>
#include <shogun/distance/Distance.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

using namespace shogun;

DistanceMachine::DistanceMachine() : Machine()
{
    init();
}

DistanceMachine::~DistanceMachine()
{
}

void DistanceMachine::init()
{
    distance = nullptr;
    SG_ADD(&distance, "distance", "Distance to use", ParameterProperties::HYPER);
}

void DistanceMachine::distances_lhs(SGVector<float64_t>& result, index_t idx_a1, index_t idx_a2, index_t idx_b)
{
    ASSERT(result)

    // Serial implementation
    for (index_t i = idx_a1; i <= idx_a2; i++)
    {
        result[i] = distance->distance(idx_b, i);
    }
}

void DistanceMachine::distances_rhs(SGVector<float64_t>& result, index_t idx_b1, index_t idx_b2, index_t idx_a)
{
    ASSERT(result)

    // Serial implementation
    for (index_t i = idx_b1; i <= idx_b2; i++)
    {
        result[i] = distance->distance(i, idx_a);
    }
}

std::shared_ptr<MulticlassLabels> DistanceMachine::apply_multiclass(std::shared_ptr<Features> data)
{
    if (!data)
    {
        // call apply on complete right-hand side
        data = distance->get_rhs();
    }

    auto result = std::make_shared<MulticlassLabels>(data->get_num_vectors());

    // Serial implementation
    for (index_t i = 0; i < data->get_num_vectors(); ++i)
    {
        result->set_label(i, apply_one(i));
    }

    return result;
}

float64_t DistanceMachine::apply_one(int32_t num)
{
    // number of clusters
    auto lhs = distance->get_lhs();
    int32_t num_clusters = lhs->get_num_vectors();

    // calculate distances to all cluster centers
    SGVector<float64_t> dists(num_clusters);
    distances_lhs(dists, 0, num_clusters - 1, num);

    // find cluster index with smallest distance
    float64_t result = dists.vector[0];
    index_t best_index = 0;
    for (index_t i = 1; i < num_clusters; ++i)
    {
        if (dists[i] < result)
        {
            result = dists[i];
            best_index = i;
        }
    }

    // implicit cast
    return best_index;
}

void DistanceMachine::set_distance(std::shared_ptr<Distance> d)
{
    distance = std::move(d);
}

std::shared_ptr<Distance> DistanceMachine::get_distance() const
{
    return distance;
}
