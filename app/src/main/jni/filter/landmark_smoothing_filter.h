#ifndef LANDMARK_SMOOTHING_H
#define LANDMARK_SMOOTHING_H

#include "relative_velocity_filter.h"
#include "one_euro_filter.h"
#include "time_stamp.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <iterator>


using NormalizedLandmark = std::tuple<float,float,float>;
using NormalizedLandmarkList = std::vector<NormalizedLandmark>;

using Normalized2DLandmark = std::pair<float,float>;
using Normalized2DLandmarkList = std::vector<Normalized2DLandmark>;

class VelocityFilter {
public:
	VelocityFilter(int window_size, float velocity_scale,
		float min_allowed_object_scale, int target_fps)
		: target_fps_(target_fps),
		window_size_(window_size),
		velocity_scale_(velocity_scale),
		min_allowed_object_scale_(min_allowed_object_scale) {}

	int Reset();

	int Apply(const NormalizedLandmarkList& in_landmarks,
		const std::pair<int, int>& image_size,
		const TimeStamp& timestamp,
		NormalizedLandmarkList* out_landmarks);

	int Apply2D(const Normalized2DLandmarkList& in_landmarks,
		const std::pair<int, int>& image_size,
		const TimeStamp& timestamp,
		Normalized2DLandmarkList* out_landmarks);

private:
	
	int InitializeFiltersIfEmpty(const size_t n_landmarks);
	static bool isValidLandMark(const NormalizedLandmark& m);
	static bool isValid2DLandMark(const Normalized2DLandmark& m);

	// desired fps
	int target_fps_;
	int window_size_;
	float velocity_scale_;
	float min_allowed_object_scale_;

	std::vector<RelativeVelocityFilter> x_filters_;
	std::vector<RelativeVelocityFilter> y_filters_;
	std::vector<RelativeVelocityFilter> z_filters_;
}; 


class OneEuroFilterImpl {
public:
	OneEuroFilterImpl(double frequency, double min_cutoff,double beta,
		double derivate_cutoff,float min_allowed_object_scale, 
		bool disable_value_scaling)
		: frequency_(frequency),
		min_cutoff_(min_cutoff),
		beta_(beta),
		derivate_cutoff_(derivate_cutoff),
		min_allowed_object_scale_(min_allowed_object_scale),
		disable_value_scaling_(disable_value_scaling) {}

	int Reset();

	int Apply(const NormalizedLandmarkList& in_landmarks,
		const std::pair<int, int>& image_size,
		const TimeStamp& timestamp,
		NormalizedLandmarkList* out_landmarks);

	int Apply2D(const Normalized2DLandmarkList& in_landmarks,
		const std::pair<int, int>& image_size,
		const TimeStamp& timestamp,
		Normalized2DLandmarkList* out_landmarks);

private:
	double frequency_;
	double min_cutoff_;
	double beta_;
	double derivate_cutoff_;
	double min_allowed_object_scale_;
	bool disable_value_scaling_;

	int InitializeFiltersIfEmpty(const size_t n_landmarks);
	static bool isValidLandMark(const NormalizedLandmark& m);
	static bool isValid2DLandMark(const Normalized2DLandmark& m);


	std::vector<OneEuroFilter> x_filters_;
	std::vector<OneEuroFilter> y_filters_;
	std::vector<OneEuroFilter> z_filters_;
};

#endif //LANDMARK_SMOOTHING_H