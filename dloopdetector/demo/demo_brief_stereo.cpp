/**
 * File: demo_brief.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <string>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines BriefVocabulary
#include <DLoopDetector/DLoopDetector.h> // defines BriefLoopDetector
#include <DVision/DVision.h> // Brief extractor

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "demoDetectorStereo.h"

// ROS Integration
#include <ros/ros.h>

using namespace DLoopDetector;

// ----------------------------------------------------------------------------

// static const char *VOC_FILE = "/home/dhl/git/catkin_ws/resources/brief_k10L6.voc.gz";
static const char *POSE_FILE="";
static const char *IMAGE_DIR="";
// static const int IMAGE_W = 640; // image size
// static const int IMAGE_H = 480;
// static const char *BRIEF_PATTERN_FILE = "/home/dhl/git/catkin_ws/resources/brief_pattern.yml";

static const int BRIEF_BIT_LENGTH = 256;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// This functor extracts BRIEF descriptors in the required format
class BriefExtractor: public FeatureExtractor<DBoW2::FBrief::TDescriptor>
{
public:

  // Proper initialisation of BRIEF descriptor
  BriefExtractor() : m_brief(BRIEF_BIT_LENGTH) {}

  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    std::vector<cv::KeyPoint> &keys, std::vector<DBoW2::FBrief::TDescriptor> &descriptors) const;

  /**
   * Creates the brief extractor with the given pattern file
   * @param pattern_file
   */
  BriefExtractor(const std::string &pattern_file);

private:

  /// BRIEF descriptor extractor
  DVision::BRIEF m_brief;  // default: BRIEF(int nbits = 256, int patch_size = 48, Type type = RANDOM_CLOSE);
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

int main(int argc, char **argv)
{
  ros::init(argc, argv, "demo_brief");

	ros::NodeHandle local_nh("~");

  std::string VOC_FILE, BRIEF_PATTERN_FILE;
  int IMAGE_W, IMAGE_H;
  local_nh.param("IMAGE_W",IMAGE_W, 1280 );
  local_nh.param("IMAGE_H",IMAGE_H, 720 );
  local_nh.param<std::string>("VOC_FILE",VOC_FILE, "./VOC_FILE" );
  local_nh.param<std::string>("BRIEF_PATTERN_FILE",BRIEF_PATTERN_FILE, "./BRIEF_PATTERN_FILE" );

  // prepares the demo
  demoDetector<BriefVocabulary, BriefLoopDetector, DBoW2::FBrief> 
    demo(VOC_FILE, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H);
  
  try 
  {
    // run the demo with the given functor to extract features
    BriefExtractor extractor(BRIEF_PATTERN_FILE);
    demo.run("BRIEF", extractor);
  }
  catch(const std::string &ex)
  {
    std::cout << "Error: " << ex << std::endl;
  }

  return 0;
}

// ----------------------------------------------------------------------------

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary
  
  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw std::string("Could not open file ") + pattern_file;
  
  std::vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;
  
  m_brief.importPairs(x1, y1, x2, y2);
}

// ----------------------------------------------------------------------------

void BriefExtractor::operator() (const cv::Mat &im, 
  std::vector<cv::KeyPoint> &keys, std::vector<DBoW2::FBrief::TDescriptor> &descriptors) const
{
  // extract FAST keypoints with opencv
  const int fast_th = 20; // corner detector response threshold
  cv::FAST(im, keys, fast_th, true);
  
  // compute their BRIEF descriptor
  m_brief.compute(im, keys, descriptors);
}

// ----------------------------------------------------------------------------

