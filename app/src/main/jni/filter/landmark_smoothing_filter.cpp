#include "landmark_smoothing_filter.h"

static float GetObjectScale(const NormalizedLandmarkList& landmarks, int image_width,
                            int image_height) {
    const auto& lm_min_x = std::min_element(
                                            landmarks.begin(), landmarks.end(),
                                            [](const NormalizedLandmark& a, const NormalizedLandmark& b) { return std::get<0>(a) < std::get<0>(b); });
    const auto& lm_max_x = std::max_element(
                                            landmarks.begin(), landmarks.end(),
                                            [](const NormalizedLandmark& a, const NormalizedLandmark& b) { return std::get<0>(a) > std::get<0>(b); });
    
    if (landmarks.size() <= 0)
        return 0;
    
    const float x_min = std::get<0>(*lm_min_x);
    const float x_max = std::get<0>(*lm_max_x);
    
    const auto& lm_min_y = std::min_element(
                                            landmarks.begin(), landmarks.end(),
                                            [](const NormalizedLandmark& a, const NormalizedLandmark& b) { return std::get<1>(a) < std::get<1>(b); });
    const auto& lm_max_y = std::max_element(
                                            landmarks.begin(), landmarks.end(),
                                            [](const NormalizedLandmark& a, const NormalizedLandmark& b) { return std::get<1>(a) > std::get<1>(b); });
    
    const float y_min = std::get<1>(*lm_min_y);
    const float y_max = std::get<1>(*lm_max_y);
    
    const float object_width = (x_max - x_min) * image_width;
    const float object_height = (y_max - y_min) * image_height;
    
    return (object_width + object_height) / 2.0f;
}

static float GetObjectScale(Normalized2DLandmarkList& landmarks, int image_width,
                            int image_height) 
{
    

	std::sort(landmarks.begin(), landmarks.end(), [](const Normalized2DLandmark & a, const Normalized2DLandmark & b) { return a.first > b.first; });
	const auto& lm_max_x = &landmarks[0];
	const auto& lm_min_x = &landmarks[landmarks.size()-1];
	//const auto& lm_min_x = std::min_element(
	//                                        landmarks.begin(), landmarks.end(),
	//                                        [](const Normalized2DLandmark& a, const Normalized2DLandmark& b) { return a.first < b.first; });
	/*const auto& lm_max_x = std::max_element(
										landmarks.begin(), landmarks.end(),
										[](const Normalized2DLandmark& a, const Normalized2DLandmark& b) { return a.first > b.first; });*/
    if (landmarks.size() <= 0)
        return 0;
    
    const float x_min = lm_min_x->first;
    const float x_max = lm_max_x->first;
    
	std::sort(landmarks.begin(), landmarks.end(), [](const Normalized2DLandmark & a, const Normalized2DLandmark & b) { return a.second > b.second; });
	const auto& lm_max_y = &landmarks[0];
	const auto& lm_min_y = &landmarks[landmarks.size() - 1];
    /*const auto& lm_min_y = std::min_element(
                                            landmarks.begin(), landmarks.end(),
                                            [](const Normalized2DLandmark& a, const Normalized2DLandmark& b) { return a.second < b.second; });
    const auto& lm_max_y = std::max_element(
                                            landmarks.begin(), landmarks.end(),
                                            [](const Normalized2DLandmark& a, const Normalized2DLandmark& b) { return a.second > b.second; });*/
    
    const float y_min = lm_min_y->second;
    const float y_max = lm_max_y->second;
    
    const float object_width = (x_max - x_min) * image_width;
    const float object_height = (y_max - y_min) * image_height;
    
    return (object_width + object_height) / 2.0f;
}

int VelocityFilter::Reset() {
    x_filters_.clear();
    y_filters_.clear();
    z_filters_.clear();
    return 0;
}

bool VelocityFilter::isValidLandMark(const NormalizedLandmark& m) {
    bool valid = (std::get<0>(m) >= 0 &&
                  std::get<1>(m) >= 0 &&
                  std::get<2>(m) >= 0);
    return valid;
}

bool VelocityFilter::isValid2DLandMark(const Normalized2DLandmark& m) {
    bool valid = (m.first >= 0) && (m.second >= 0);
    return valid;
}

int VelocityFilter::Apply(const NormalizedLandmarkList& in_landmarks,
                                     const std::pair<int, int>& image_size,
                                     const TimeStamp& timestamp,
                                     NormalizedLandmarkList* out_landmarks) {
    int image_width;
    int image_height;
    std::tie(image_height, image_width) = image_size;

    NormalizedLandmarkList filterd;
    std::copy_if(in_landmarks.begin(), in_landmarks.end(), std::back_inserter(filterd), isValidLandMark);
    const float object_scale = GetObjectScale(filterd, image_width, image_height);
    if (object_scale < min_allowed_object_scale_) {
        *out_landmarks = in_landmarks;
        return 0;
    }
    const float value_scale = 1.0f / object_scale;

    auto status = InitializeFiltersIfEmpty(in_landmarks.size());
	if (status != 0)
		return -1;

    for (int i = 0; i < in_landmarks.size(); ++i) {
        const NormalizedLandmark& in_landmark = in_landmarks[i];
        
        if (!isValidLandMark(in_landmark)) {
            out_landmarks->push_back(in_landmark);
            continue;
        }
        
        float out_x = x_filters_[i].Apply(timestamp, value_scale,
                                          std::get<0>(in_landmark) * image_width) / image_width;
        float out_y = y_filters_[i].Apply(timestamp, value_scale,
                                          std::get<1>(in_landmark) * image_height) / image_height;
        float out_z = z_filters_[i].Apply(timestamp, value_scale,
                                          std::get<2>(in_landmark) * image_width) / image_width;
        
        NormalizedLandmark out_landmark = std::make_tuple(out_x, out_y, out_z);
        out_landmarks->push_back(std::move(out_landmark));
    }
    
    return 0;
}

int VelocityFilter::Apply2D(const Normalized2DLandmarkList& in_landmarks,
                                       const std::pair<int, int>& image_size,
                                       const TimeStamp& timestamp,
                                       Normalized2DLandmarkList* out_landmarks) {
    int image_width;
    int image_height;
    std::tie(image_height, image_width) = image_size;
    
    Normalized2DLandmarkList filterd;
    std::copy_if(in_landmarks.begin(), in_landmarks.end(), std::back_inserter(filterd), isValid2DLandMark);
    const float object_scale = GetObjectScale(filterd, image_width, image_height);
    if (object_scale < min_allowed_object_scale_) {
        *out_landmarks = in_landmarks;
        return 0;
    }
    const float value_scale = 1.0f / object_scale;
    
    // Initialize filters once.
    auto status = InitializeFiltersIfEmpty(in_landmarks.size());
	if (status != 0)
		return -1;
    
    // Filter landmarks. Every axis of every landmark is filtered separately.
    for (int i = 0; i < in_landmarks.size(); ++i) {
        const Normalized2DLandmark& in_landmark = in_landmarks[i];
        
        if (!isValid2DLandMark(in_landmark)) {
            out_landmarks->push_back(in_landmark);
            continue;
        }
        
        float out_x = x_filters_[i].Apply(timestamp, value_scale,
                                          in_landmark.first * image_width) / image_width;
        float out_y = y_filters_[i].Apply(timestamp, value_scale,
                                          in_landmark.second * image_height) / image_height;
        
        Normalized2DLandmark out_landmark = std::make_pair(out_x, out_y);
        out_landmarks->push_back(std::move(out_landmark));
    }
    
    return 0;
}

int VelocityFilter::InitializeFiltersIfEmpty(const size_t n_landmarks) {
    if (!x_filters_.empty()) 
	{
		if (x_filters_.size() != n_landmarks)
			return -1;
		if (y_filters_.size() != n_landmarks)
			return -1;
		if (z_filters_.size() != n_landmarks)
			return -1;
        return 0;
    }
    
    x_filters_.resize(n_landmarks,
                      RelativeVelocityFilter(window_size_, velocity_scale_, target_fps_));
    y_filters_.resize(n_landmarks,
                      RelativeVelocityFilter(window_size_, velocity_scale_, target_fps_));
    z_filters_.resize(n_landmarks,
                      RelativeVelocityFilter(window_size_, velocity_scale_, target_fps_));
    
    return 0;
}

int OneEuroFilterImpl::Reset() {
	x_filters_.clear();
	y_filters_.clear();
	z_filters_.clear();
	return 0;
}

bool OneEuroFilterImpl::isValidLandMark(const NormalizedLandmark& m) {
	bool valid = (std::get<0>(m) >= 0 &&
		std::get<1>(m) >= 0 &&
		std::get<2>(m) >= 0);
	return valid;
}

bool OneEuroFilterImpl::isValid2DLandMark(const Normalized2DLandmark & m) {
	bool valid = (m.first >= 0) && (m.second >= 0);
	return valid;
}

int OneEuroFilterImpl::Apply(const NormalizedLandmarkList & in_landmarks,
	const std::pair<int, int> & image_size,
	const TimeStamp & timestamp,
	NormalizedLandmarkList * out_landmarks) {
	int image_width;
	int image_height;
	std::tie(image_height, image_width) = image_size;

	NormalizedLandmarkList filterd;
	std::copy_if(in_landmarks.begin(), in_landmarks.end(), std::back_inserter(filterd), isValidLandMark);
	const float object_scale = GetObjectScale(filterd, image_width, image_height);
	if (object_scale < min_allowed_object_scale_) {
		*out_landmarks = in_landmarks;
		return 0;
	}
	const float value_scale = 1.0f / object_scale;

	auto status = InitializeFiltersIfEmpty(in_landmarks.size());
	if (status != 0)
		return -1;

	for (int i = 0; i < in_landmarks.size(); ++i) {
		const NormalizedLandmark& in_landmark = in_landmarks[i];

		if (!isValidLandMark(in_landmark)) {
			out_landmarks->push_back(in_landmark);
			continue;
		}

		float out_x = x_filters_[i].Apply(timestamp, value_scale,
			std::get<0>(in_landmark) * image_width) / image_width;
		float out_y = y_filters_[i].Apply(timestamp, value_scale,
			std::get<1>(in_landmark) * image_height) / image_height;
		float out_z = z_filters_[i].Apply(timestamp, value_scale,
			std::get<2>(in_landmark) * image_width) / image_width;

		NormalizedLandmark out_landmark = std::make_tuple(out_x, out_y, out_z);
		out_landmarks->push_back(std::move(out_landmark));
	}

	return 0;
}

int OneEuroFilterImpl::Apply2D(const Normalized2DLandmarkList & in_landmarks,
	const std::pair<int, int> & image_size,
	const TimeStamp & timestamp,
	Normalized2DLandmarkList * out_landmarks) {
	int image_width;
	int image_height;
	std::tie(image_height, image_width) = image_size;

	Normalized2DLandmarkList filterd;
	std::copy_if(in_landmarks.begin(), in_landmarks.end(), std::back_inserter(filterd), isValid2DLandMark);
	const float object_scale = GetObjectScale(filterd, image_width, image_height);
	if (object_scale < min_allowed_object_scale_) {
		*out_landmarks = in_landmarks;
		return 0;
	}
	const float value_scale = 1.0f / object_scale;

	// Initialize filters once.
	auto status = InitializeFiltersIfEmpty(in_landmarks.size());
	if (status != 0)
		return -1;

	// Filter landmarks. Every axis of every landmark is filtered separately.
	for (int i = 0; i < in_landmarks.size(); ++i) {
		const Normalized2DLandmark& in_landmark = in_landmarks[i];

		if (!isValid2DLandMark(in_landmark)) {
			out_landmarks->push_back(in_landmark);
			continue;
		}

		float out_x = x_filters_[i].Apply(timestamp, value_scale,
			in_landmark.first * image_width) / image_width;
		float out_y = y_filters_[i].Apply(timestamp, value_scale,
			in_landmark.second * image_height) / image_height;

		Normalized2DLandmark out_landmark = std::make_pair(out_x, out_y);
		out_landmarks->push_back(std::move(out_landmark));
	}

	return 0;
}

int OneEuroFilterImpl::InitializeFiltersIfEmpty(const size_t n_landmarks) {
	if (!x_filters_.empty())
	{
		if (x_filters_.size() != n_landmarks)
			return -1;
		if (y_filters_.size() != n_landmarks)
			return -1;
		if (z_filters_.size() != n_landmarks)
			return -1;
		return 0;
	}

	x_filters_.resize(n_landmarks,
		OneEuroFilter(frequency_, min_cutoff_, beta_, derivate_cutoff_));
	y_filters_.resize(n_landmarks,
		OneEuroFilter(frequency_, min_cutoff_, beta_, derivate_cutoff_));
	z_filters_.resize(n_landmarks,
		OneEuroFilter(frequency_, min_cutoff_, beta_, derivate_cutoff_));

	return 0;
}