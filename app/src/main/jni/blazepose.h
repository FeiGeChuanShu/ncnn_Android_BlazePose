// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef FACE_H
#define FACE_H

#include <opencv2/core/core.hpp>
#include <net.h>
#include "landmark.h"
#include "landmark_smoothing_filter.h"
#include "time_stamp.h"
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float score;
    cv::Point2f  landmarks[4];
    float  rotation;
    float  cx;
    float  cy;
    float  w;
    float  h;
    cv::Point2f  points[4];
    cv::Mat trans_image;
    std::vector<Keypoint> skeleton;
};


struct DetectRegion
{
    float score;
    cv::Point2f topleft;
    cv::Point2f btmright;
    cv::Point2f landmarks[4];
    float  rotation;
};
struct Anchor
{
    float x_center, y_center, w, h;
};

struct AnchorsParams
{
    int num_layers;
    int input_size_width;
    int input_size_height;
    float min_scale;
    float max_scale;
    float anchor_offset_x;
    float anchor_offset_y;
    std::vector<int>   strides;
    std::vector<float> aspect_ratios;

};

class BlazePose
{
public:
    BlazePose();

    int load(AAssetManager* mgr, const char* modeltype, bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<Object>& objects);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

private:
    void smoothingLandmarks(std::vector<Keypoint>& detects, int img_w,int img_h);
    const int window_size = 5;
    const float velocity_scale = 10.0;
    const float min_allowed_object_scale = 1e-6;
    std::shared_ptr<VelocityFilter> velocity_landmark_filter;

    //const float frequency = 30.0;
    //const float min_cutoff = 0.05;
    //const float beta = 80.0;
    //const float derivate_cutoff = 1.0;
    //bool disable_value_scaling = false;
    //std::shared_ptr<OneEuroFilterImpl> one_euro_landmark_filter;
private:
    ncnn::Net pose_detection;
    LandmarkDetect pose_landmark;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
    std::vector<Anchor> anchors;
};

#endif // FACE_H
