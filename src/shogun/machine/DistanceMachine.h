/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann, Yuyu Zhang,
 *          Thoralf Klein, Evan Shelhamer, Saurabh Goyal
 */

#ifndef _DISTANCE_MACHINE_H__
#define _DISTANCE_MACHINE_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <memory>

namespace shogun {
	class Distance;
	class Features;
	class MulticlassLabels;

	/** @brief A generic DistanceMachine interface.
	 *
	 * A distance machine is based on a a-priori chosen distance.
	 */
	class DistanceMachine : public Machine {
	public:
		/** default constructor */
		DistanceMachine();

		/** destructor */
		~DistanceMachine() override;

		/** set distance */
		void set_distance(std::shared_ptr<Distance> d);

		/** get distance */
		std::shared_ptr<Distance> get_distance() const;

		/** apply to multiclass labels */
		std::shared_ptr<MulticlassLabels> apply_multiclass(std::shared_ptr<Features> data);

	protected:
		/** initialize machine before training (train or apply) */
		void init() override;

		/** calculate distances to lhs */
		void distances_lhs(SGVector<float64_t>& result, index_t idx_a1, index_t idx_a2, index_t idx_b);

		/** calculate distances to rhs */
		void distances_rhs(SGVector<float64_t>& result, index_t idx_b1, index_t idx_b2, index_t idx_a);

		/** apply to one vector */
		float64_t apply_one(int32_t num);

	private:
		/** distance to use */
		std::shared_ptr<Distance> distance;
	};
}
#endif // _DISTANCE_MACHINE_H__
