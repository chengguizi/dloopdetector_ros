/**
 * File: demoDetector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

#ifndef __DEMO_DETECTOR__
#define __DEMO_DETECTOR__

#include <iostream>
#include <vector>
#include <string>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h>
#include <DLoopDetector/DLoopDetector.h>
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DVision/DVision.h>

// ROS Integration
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <image_transport/subscriber_filter.h>

#include <cv_bridge/cv_bridge.h>

// Cam Motion Estimation
#include <CamMotionEstimator/CamMotionEstimator.h>

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
	/**
	 * Extracts features
	 * @param im image
	 * @param keys keypoints extracted
	 * @param descriptors descriptors extracted
	 */
	virtual void operator()(const cv::Mat &im, 
		vector<cv::KeyPoint> &keys, vector<TDescriptor> &descriptors) const = 0;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// @param TVocabulary vocabulary class (e.g: Surf64Vocabulary)
/// @param TDetector detector class (e.g: Surf64LoopDetector)
/// @param TDescriptor descriptor class (e.g: vector<float> for SURF)
template<class TVocabulary, class TDetector, class TFeature>
/// Class to run the demo 
class demoDetector
{
public:
	typedef typename TFeature::TDescriptor TDescriptor;
	/**
	 * @param vocfile vocabulary file to load
	 * @param imagedir directory to read images from
	 * @param posefile pose file
	 * @param width image width
	 * @param height image height
	 */
	demoDetector(const std::string &vocfile, const std::string &imagedir,
		const std::string &posefile, int width, int height);
		
	~demoDetector(){}

	/**
	 * Runs the demo
	 * @param name demo name
	 * @param extractor functor to extract features
	 */
	void run(const std::string &name, 
		const FeatureExtractor<TDescriptor> &extractor);

protected:

	/**
	 * Reads the robot poses from a file
	 * @param filename file
	 * @param xs
	 * @param ys
	 */
	void readPoseFile(const char *filename, std::vector<double> &xs, 
		std::vector<double> &ys) const;

protected:

	std::string m_vocfile;
	std::string m_imagedir;
	std::string m_posefile;
	int m_width;
	int m_height;
	FeatureExtractor<TDescriptor> *extractor_;
	

	// ROS

	image_transport::SubscriberFilter left_sub_;
	image_transport::SubscriberFilter right_sub_;
	message_filters::Subscriber<sensor_msgs::CameraInfo> left_info_sub_;
	message_filters::Subscriber<sensor_msgs::CameraInfo> right_info_sub_;
	typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactPolicy;
	typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
	ExactSync exact_sync_;

	int all_received_;
	sensor_msgs::ImageConstPtr l_image_msg_;
	sensor_msgs::ImageConstPtr r_image_msg_;
	sensor_msgs::CameraInfoConstPtr l_info_msg_;
	sensor_msgs::CameraInfoConstPtr r_info_msg_;

	void dataCb(const sensor_msgs::ImageConstPtr& l_image_msg,
							const sensor_msgs::ImageConstPtr& r_image_msg,
							const sensor_msgs::CameraInfoConstPtr& l_info_msg,
							const sensor_msgs::CameraInfoConstPtr& r_info_msg
						)
	{
		all_received_++;
		l_image_msg_ = l_image_msg;
		r_image_msg_ = r_image_msg;
		l_info_msg_ = l_info_msg;
		r_info_msg_ = r_info_msg;
		// cout << "Received No." << all_received_ << "ros image" << endl;
	}
};

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TFeature>
demoDetector<TVocabulary, TDetector, TFeature>::demoDetector
	(const std::string &vocfile, const std::string &imagedir,
	const std::string &posefile, int width, int height)
	: m_vocfile(vocfile), m_imagedir(imagedir), m_posefile(posefile),
		m_width(width), m_height(height), exact_sync_(ExactPolicy(3), left_sub_, right_sub_, left_info_sub_, right_info_sub_), all_received_(0)
{}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TFeature>
void demoDetector<TVocabulary, TDetector, TFeature>::run
	(const std::string &name, const FeatureExtractor<TDescriptor> &extractor)
{
	cout << "DLoopDetector Demo - Modified by Cheng Huimin" << endl 
		<< "Dorian Galvez-Lopez" << endl
		<< "http://doriangalvez.com" << endl << endl;
	
	// Set loop detector parameters
	typename TDetector::Parameters params(m_height, m_width);
	
	// Parameters given by default are:
	// use nss = true
	// alpha = 0.3
	// k = 3
	// geom checking = GEOM_DI
	// di levels = 0
	
	// We are going to change these values individually:
	params.use_nss = true; // use normalized similarity score instead of raw score
	params.alpha = 0.3; // nss threshold
	params.k = 1; // a loop must be consistent with 1 previous matches
	params.geom_check = GEOM_DI; // use direct index for geometrical checking
	params.di_levels = 2; // use two direct index levels
	
	// To verify loops you can select one of the next geometrical checkings:
	// GEOM_EXHAUSTIVE: correspondence points are computed by comparing all
	//    the features between the two images.
	// GEOM_FLANN: as above, but the comparisons are done with a Flann structure,
	//    which makes them faster. However, creating the flann structure may
	//    be slow.
	// GEOM_DI: the direct index is used to select correspondence points between
	//    those features whose vocabulary node at a certain level is the same.
	//    The level at which the comparison is done is set by the parameter
	//    di_levels:
	//      di_levels = 0 -> features must belong to the same leaf (word).
	//         This is the fastest configuration and the most restrictive one.
	//      di_levels = l (l < L) -> node at level l starting from the leaves.
	//         The higher l, the slower the geometrical checking, but higher
	//         recall as well.
	//         Here, L stands for the depth levels of the vocabulary tree.
	//      di_levels = L -> the same as the exhaustive technique.
	// GEOM_NONE: no geometrical checking is done.
	//
	// In general, with a 10^6 vocabulary, GEOM_DI with 2 <= di_levels <= 4 
	// yields the best results in recall/time.
	// Check the T-RO paper for more information.
	//
	
	// Load the vocabulary to use
	cout << "Loading " << name << " vocabulary..." << endl;
	TVocabulary voc(m_vocfile);
	
	// Initiate loop detector with the vocabulary 
	cout << "Processing sequence..." << endl;
	TDetector detector(voc, params); // typedef TemplatedLoopDetector
	
	// Process images
	vector<cv::KeyPoint> keys, keys_right;
	vector<TDescriptor> descriptors, descriptors_right;

	// load image filenames  
	// vector<string> filenames = 
	//   DUtils::FileFunctions::Dir(m_imagedir.c_str(), ".png", true);
	
	// load robot poses
	// vector<double> xs, ys;
	// readPoseFile(m_posefile.c_str(), xs, ys);
	
	// we can allocate memory for the expected number of images
	cout << "Allocating memory for detector" << endl;
	// detector.allocate(filenames.size());
	detector.allocate(1000);

	// prepare visualization windows
	DUtilsCV::GUI::tWinHandler win = "Current image";
	// DUtilsCV::GUI::tWinHandler winplot = "Trajectory";
	
	// DUtilsCV::Drawing::Plot::Style normal_style(2); // thickness
	// DUtilsCV::Drawing::Plot::Style loop_style('r', 2); // color, thickness
	
	// DUtilsCV::Drawing::Plot implot(240, 320,
	//   - *std::max_element(xs.begin(), xs.end()),
	//   - *std::min_element(xs.begin(), xs.end()),
	//   *std::min_element(ys.begin(), ys.end()),
	//   *std::max_element(ys.begin(), ys.end()), 20);
	
	// prepare profiler to measure times
	DUtils::Profiler profiler;


	// ROS INITIALISATION

	// Read local parameters
	ros::NodeHandle local_nh("~");
	// Resolve topic names
	ros::NodeHandle nh;

	std::string left_topic, right_topic;
	std::string left_info_topic, right_info_topic;
	local_nh.param<std::string>("left_image",left_topic, "/stereo/left/image_rect_raw" );
	local_nh.param<std::string>("right_image",right_topic, "/stereo/right/image_rect_raw" );
	local_nh.param<std::string>("left_camerainfo",left_info_topic, "/stereo/left/camera_info" );
	local_nh.param<std::string>("right_camerainfo",right_info_topic, "/stereo/right/camera_info" );

	// Subscribe to four input topics.
	ROS_INFO("viso2_ros: Subscribing to:\n\t* %s\n\t* %s\n\t* %s\n\t* %s", 
				left_topic.c_str(), right_topic.c_str(),
				left_info_topic.c_str(), right_info_topic.c_str());

	image_transport::ImageTransport it(nh);
	image_transport::TransportHints hints("raw",ros::TransportHints().tcpNoDelay());
	left_sub_.subscribe(it, left_topic, 1, hints); // http://docs.ros.org/diamondback/api/image_transport/html/classimage__transport_1_1TransportHints.html
	right_sub_.subscribe(it, right_topic, 1, hints);
	left_info_sub_.subscribe(nh, left_info_topic, 1,  ros::TransportHints().tcpNoDelay());
	right_info_sub_.subscribe(nh, right_info_topic, 1,  ros::TransportHints().tcpNoDelay());

	exact_sync_.registerCallback(boost::bind(&demoDetector::dataCb, this, _1, _2, _3, _4));
	
	int count = 0;
	int db_size = 0;

	
	CamMotionEstimator<FBrief::TDescriptor, TFeature> camMotionEstimator(m_width, m_height);

	// go
	while(true)
	{
		
		
		// get image from ROS
		while (ros::ok() && all_received_/10 <= db_size)
		{
			ros::spinOnce();
			ros::Duration(0.01).sleep();
		}

		if(!ros::ok())
		{
			cout << "ROS Shutting down" << endl;
			break;
		}
		
		cout << "Adding image " << db_size << endl;

		// Retrive images from ROS
		uint8_t *l_image_data, r_image_data;
		cv_bridge::CvImageConstPtr l_cv_ptr, r_cv_ptr;
		l_cv_ptr = cv_bridge::toCvShare(l_image_msg_, sensor_msgs::image_encodings::MONO8);
		r_cv_ptr = cv_bridge::toCvShare(r_image_msg_, sensor_msgs::image_encodings::MONO8);

		ROS_ASSERT( m_width == (int)l_image_msg_->width && m_height == (int)l_image_msg_->height);
		cv::Mat im(cv::Size(m_width, m_height), CV_8UC1, (void *)l_cv_ptr->image.data, cv::Mat::AUTO_STEP);
		cv::Mat im_right(cv::Size(m_width, m_height), CV_8UC1, (void *)r_cv_ptr->image.data, cv::Mat::AUTO_STEP);

		
		// get features
		profiler.profile("features");
		extractor(im, keys, descriptors);
		extractor(im_right, keys_right, descriptors_right);
		profiler.stop();

		// show image
		cv::Mat outIm(cv::Size(m_width,m_height),CV_8UC1);
		cv::drawKeypoints(im,keys,outIm);
		DUtilsCV::GUI::showImage(outIm, true, &win, 10);
				
		// add image to the collection and check if there is some loop
		typename TDetector::DetectionResult result;
		
		profiler.profile("detection");
		detector.detectLoop(keys, descriptors, result, keys_right, descriptors_right); // db_size + 1 for each of the detectLoop operation
		db_size++;
		profiler.stop();
		
		if(result.detection())
		{
			cout << "- Loop found query image " << result.query << " with match image " << result.match << "!" << endl;

			camMotionEstimator.pushBackData (result.match_keys.main, result.match_keys.right,
											result.query_keys.main, result.query_keys.right, 
											result.match_descriptors.main, result.match_descriptors.right,
											result.query_descriptors.main, result.query_descriptors.right);
			
			if (camMotionEstimator.matchFeaturesQuad())
			{
				cout << "matchFeaturesQuad() Done." << endl;
			}else
				cerr << "matchFeaturesQuad() Fails." << endl;

			//CamMotionEstimator<TDescriptor>::computeRTfromStereo(result.query_keys, result.match_keys, result.query_descriptors,  result.match_descriptors);
			++count;
		}
		else
		{
			cout << "- No loop: ";
			switch(result.status)
			{
				case CLOSE_MATCHES_ONLY:
					cout << "All the images in the database are very recent" << endl;
					break;
					
				case NO_DB_RESULTS:
					cout << "There are no matches against the database (few features in"
						" the image?)" << endl;
					break;
					
				case LOW_NSS_FACTOR:
					cout << "Little overlap between this image and the previous one"
						<< endl;
					break;
						
				case LOW_SCORES:
					cout << "No match reaches the score threshold (alpha: " <<
						params.alpha << ")" << endl;
					break;
					
				case NO_GROUPS:
					cout << "Not enough close matches to create groups. "
						<< "Best candidate: " << result.match << endl;
					break;
					
				case NO_TEMPORAL_CONSISTENCY:
					cout << "No temporal consistency (k: " << params.k << "). "
						<< "Best candidate: " << result.match << endl;
					break;
					
				case NO_GEOMETRICAL_CONSISTENCY:
					cout << "No geometrical consistency. Best candidate: " 
						<< result.match << endl;
					break;
					
				default:
					break;
			}
		}
		cout << " Loop detection (mean): " << profiler.getMeanTime("detection") * 1e3 << " ms/image" ;
		cout << endl;
		
		// // show trajectory
		// if(i > 0)
		// {
		//   if(result.detection())
		//     implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], loop_style);
		//   else
		//     implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], normal_style);
			
		//   DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 10); 
		// }
	}
	
	if(count == 0)
	{
		cout << "No loops found in this image sequence" << endl;
	}
	else
	{
		cout << count << " loops found in this image sequence!" << endl;
	} 

	cout << endl << "Execution time:" << endl
		<< " - Feature computation : (mean) " << profiler.getMeanTime("features") * 1e3
		<< " ms/image, (max)" << profiler.getMaxTime("features") * 1e3
		<< " ms/image" << endl
		<< " - Loop detection (mean): " << profiler.getMeanTime("detection") * 1e3
		<< " ms/image, (max)" << profiler.getMaxTime("detection") * 1e3
		<< " ms/image" << endl;

	cout << endl << "Press a key to finish..." << endl;
	// DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 0);
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TFeature>
void demoDetector<TVocabulary, TDetector, TFeature>::readPoseFile
	(const char *filename, std::vector<double> &xs, std::vector<double> &ys)
	const
{
	xs.clear();
	ys.clear();
	
	fstream f(filename, ios::in);
	
	string s;
	double ts, x, y, t;
	while(!f.eof())
	{
		getline(f, s);
		if(!f.eof() && !s.empty())
		{
			sscanf(s.c_str(), "%lf, %lf, %lf, %lf", &ts, &x, &y, &t);
			xs.push_back(x);
			ys.push_back(y);
		}
	}
	
	f.close();
}

// ---------------------------------------------------------------------------

#endif

