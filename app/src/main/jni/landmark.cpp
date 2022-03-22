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

#include "landmark.h"

#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"



int LandmarkDetect::load(AAssetManager* mgr, const char* modeltype, bool use_gpu)
{
    landmark.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    landmark.opt = ncnn::Option();

#if NCNN_VULKAN
    landmark.opt.use_vulkan_compute = use_gpu;
#endif

    landmark.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    landmark.load_param(mgr, parampath);
    landmark.load_model(mgr, modelpath);


    return 0;
}
static void getOriginalCoord(const std::vector<Keypoint>& keypoints, const cv::Mat& trans_mat, std::vector<Keypoint>& trans_points)
{
    cv::Mat trans_mat_inv;
    cv::invertAffineTransform(trans_mat, trans_mat_inv);

    for (const auto& kpt : keypoints)
    {
        float x = kpt.x;
        float y = kpt.y;

        x = kpt.x * trans_mat_inv.at<double>(0, 0) + kpt.y * trans_mat_inv.at<double>(0, 1) + trans_mat_inv.at<double>(0, 2);
        y = kpt.x * trans_mat_inv.at<double>(1, 0) + kpt.y * trans_mat_inv.at<double>(1, 1) + trans_mat_inv.at<double>(1, 2);

        trans_points.push_back(Keypoint(x,y,kpt.visibility));
    }

}
static void decodeKeypoints(const ncnn::Mat& id_3d, int img_w, int img_h, std::vector<Keypoint>& keypoints)
{
    const int num_kpt = 39;
    const int num_kpt_offset = id_3d.total() / num_kpt;
    for (size_t i = 0; i < num_kpt; i++)
    {
        float pt_x = id_3d[i * num_kpt_offset] * img_w / 256;
        float pt_y = id_3d[i * num_kpt_offset + 1] * img_h / 256;
        //1.for whole body,all points is visible
        bool visibility = true;
        //2.for half body or occluded,make some points invisible
        //bool visibility = id_3d[i*num_kpt_offset+3] > 0.1 && id_3d[i*num_kpt_offset+4] > 0.3 ? true : false;
        keypoints.push_back(Keypoint(pt_x, pt_y, visibility));
    }
}


float LandmarkDetect::detect(const cv::Mat& rgb,const cv::Mat& trans_mat, std::vector<Keypoint> &landmarks)
{
    cv::Mat input = rgb.clone();

    ncnn::Mat in = ncnn::Mat::from_pixels(input.data, ncnn::Mat::PIXEL_RGB, input.cols, input.rows);

    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in.substract_mean_normalize(0, norm_vals);
    ncnn::Mat seg, heatmap, poseflag, id_3d, world_3d;
    ncnn::Extractor ex = landmark.create_extractor();

    ex.input("input", in);
    ex.extract("poseflag", poseflag);
    if(poseflag[0] < 0.5)
        return -1.f;
    ex.extract("ld_3d", id_3d);
    //ex.extract("seg", seg);
    std::vector<Keypoint> keypoints;
    decodeKeypoints(id_3d, rgb.cols, rgb.rows, keypoints);

    getOriginalCoord(keypoints, trans_mat, landmarks);
    return poseflag[0];
}

