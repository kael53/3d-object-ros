// yolox_2d_to_3d_objects_node.cpp (updated to publish Detection3DArray)
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>

#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <vision_msgs/msg/bounding_box3_d.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <geometry_msgs/msg/vector3.hpp>

#include "bboxes_ex_msgs/msg/bounding_box.hpp"
#include "bboxes_ex_msgs/msg/bounding_boxes.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <memory>

struct Detection {
  std::string class_name;
  float confidence;
  int x_min, y_min, x_max, y_max;
};

class YoloxTo3DNode : public rclcpp::Node {
public:
  YoloxTo3DNode() : Node("object_3d_ros_node") {

    RCLCPP_INFO(this->get_logger(), "3D Object Detection Node Initialized");
    
    rgb_sub_.subscribe(this, "/camera/color/image_raw");
    depth_sub_.subscribe(this, "/camera/aligned_depth_to_color/image_raw");
    info_sub_.subscribe(this, "/camera/color/camera_info");

    detections_sub_ = this->create_subscription<bboxes_ex_msgs::msg::BoundingBoxes>("/yolox/bounding_boxes", 10,
        std::bind(&YoloxTo3DNode::detectionsCallback, this, std::placeholders::_1));


    sync_.reset(new Sync(SyncPolicy(10), rgb_sub_, depth_sub_, info_sub_));
    sync_->registerCallback(std::bind(&YoloxTo3DNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/objects_3d", 10);
    detection_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("/detections_3d", 10);

    allowed_classes_ = {"cup", "bottle", "book"};
  }

private:
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo> SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;

  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> info_sub_;
  std::shared_ptr<Sync> sync_;

  rclcpp::Subscription<bboxes_ex_msgs::msg::BoundingBoxes>::SharedPtr detections_sub_;
  std::vector<Detection> latest_detections_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_pub_;

  std::vector<std::string> allowed_classes_;

  void imageCallback(
      const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr &info_msg) {

    if (latest_detections_.empty()) {
      RCLCPP_WARN(this->get_logger(), "No detections received yet.");
      return;
    }
    if (!rgb_msg || !depth_msg || !info_msg) {
      RCLCPP_ERROR(this->get_logger(), "Received null message(s).");
      return;
    }
    if ((rgb_msg->height != depth_msg->height) || (rgb_msg->width != depth_msg->width)) {
      RCLCPP_ERROR(this->get_logger(), "Image sizes are different.");
      return;
    }

    cv::Mat rgb = cv_bridge::toCvShare(rgb_msg, "bgr8")->image;
    cv::Mat depth = cv_bridge::toCvShare(depth_msg)->image;

    float fx = info_msg->k[0];
    float fy = info_msg->k[4];
    float cx = info_msg->k[2];
    float cy = info_msg->k[5];

    std::vector<Detection> detections = latest_detections_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr total_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    vision_msgs::msg::Detection3DArray detection_array;
    detection_array.header = rgb_msg->header;

    for (const auto &det : detections) {
      if (std::find(allowed_classes_.begin(), allowed_classes_.end(), det.class_name) == allowed_classes_.end()) {
        RCLCPP_DEBUG(this->get_logger(), "Skipping detection: %s", det.class_name.c_str());
        continue;
      }
      if (det.confidence < 0.3) {
        RCLCPP_DEBUG(this->get_logger(), "Skipping detection with low confidence: %s (%.2f)", det.class_name.c_str(), det.confidence);
        continue;
      }

      cv::Mat roi = depth(cv::Rect(det.x_min, det.y_min, det.x_max - det.x_min, det.y_max - det.y_min));
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

      for (int v = 0; v < roi.rows; ++v) {
        for (int u = 0; u < roi.cols; ++u) {
          float Z = roi.at<float>(v, u);
          if (Z <= 0.0 || Z > 3.0) continue;
          float X = (u + det.x_min - cx) * Z / fx;
          float Y = (v + det.y_min - cy) * Z / fy;

          pcl::PointXYZRGB point;
          point.x = X;
          point.y = Y;
          point.z = Z;
          cv::Vec3b color = rgb.at<cv::Vec3b>(v + det.y_min, u + det.x_min);
          point.r = color[2];
          point.g = color[1];
          point.b = color[0];
          cloud->points.push_back(point);
        }
      }

      float coverage = float(cloud->size()) / (roi.rows * roi.cols);
      if (coverage < 0.4) {
        RCLCPP_DEBUG(this->get_logger(), "Skipping detection with low coverage: %s (coverage: %.2f)", det.class_name.c_str(), coverage);
        continue;
      }

      pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
      sor.setInputCloud(cloud);
      sor.setMeanK(20);
      sor.setStddevMulThresh(1.0);
      sor.filter(*cloud);

      *total_cloud += *cloud;

      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cloud, centroid);

      pcl::PointXYZRGB min_pt, max_pt;
      pcl::getMinMax3D(*cloud, min_pt, max_pt);

      vision_msgs::msg::Detection3D detection;
      detection.header = rgb_msg->header;

      vision_msgs::msg::ObjectHypothesisWithPose hypo;
      hypo.id = det.class_name;
      hypo.score = det.confidence;
      hypo.pose.pose.position.x = centroid[0];
      hypo.pose.pose.position.y = centroid[1];
      hypo.pose.pose.position.z = centroid[2];
      hypo.pose.pose.orientation.w = 1.0;

      detection.results.push_back(hypo);
      detection.bbox.center = hypo.pose.pose;
      detection.bbox.size.x = max_pt.x - min_pt.x;
      detection.bbox.size.y = max_pt.y - min_pt.y;
      detection.bbox.size.z = max_pt.z - min_pt.z;

      detection_array.detections.push_back(detection);
    }

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*total_cloud, cloud_msg);
    cloud_msg.header = rgb_msg->header;
    pointcloud_pub_->publish(cloud_msg);
    detection_pub_->publish(detection_array);
  }

  void detectionsCallback(const bboxes_ex_msgs::msg::BoundingBoxes::SharedPtr msg) {
    latest_detections_.clear();
    for (const auto &d : msg->bounding_boxes) {
      RCLCPP_DEBUG(this->get_logger(), "Received detection: %s with confidence %.2f", d.class_id.c_str(), d.probability);
        if (std::find(allowed_classes_.begin(), allowed_classes_.end(), d.class_id) != allowed_classes_.end()) {
            RCLCPP_DEBUG(this->get_logger(), "Adding detection: %s", d.class_id.c_str());
            latest_detections_.emplace_back(Detection{d.class_id, d.probability, d.xmin, d.ymin, d.xmax, d.ymax});
        }
    }
  }

};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<YoloxTo3DNode>());
  rclcpp::shutdown();
  return 0;
}