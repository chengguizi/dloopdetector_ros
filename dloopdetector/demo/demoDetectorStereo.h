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

#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DLoopDetector/DLoopDetector.h>

// ROS Integration
#include <ros/ros.h>
#include <rosbag/bag.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/stereo_camera_model.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/cache.h>
#include <message_filters/sync_policies/exact_time.h>
#include <image_transport/subscriber_filter.h>

#include <cv_bridge/cv_bridge.h>
#include <tf2_eigen/tf2_eigen.h>

// Quad Matching & Cam Motion Estimation
#include <viso2_eigen/quad_matcher.h>
#include <viso2_eigen/stereo_motion_estimator.h>

using namespace DLoopDetector;

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
		std::vector<cv::KeyPoint> &keys, std::vector<TDescriptor> &descriptors) const = 0;
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


	ros::Subscriber _reset_sub;
	ros::Publisher _trajectory_pub;

	void resetCallback(const std_msgs::HeaderPtr &header){
    	ROS_WARN_STREAM("Reset occurs at " << header->stamp );
		resetPlot();
  	}

	/**
	 * @param vocfile vocabulary file to load
	 * @param imagedir directory to read images from
	 * @param posefile pose file
	 * @param width image width
	 * @param height image height
	 */
	demoDetector(const std::string &vocfile, int width, int height);
		
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
	int m_width;
	int m_height;
	FeatureExtractor<TDescriptor> *extractor_;
	

	// ROS image sequence subscribe

	image_transport::SubscriberFilter left_sub_;
	image_transport::SubscriberFilter right_sub_;
	message_filters::Subscriber<sensor_msgs::CameraInfo> left_info_sub_;
	message_filters::Subscriber<sensor_msgs::CameraInfo> right_info_sub_;
	typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactPolicy;
	typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
	ExactSync exact_sync_;

	int all_received_, pose_received_;
	sensor_msgs::ImageConstPtr l_image_msg_;
	sensor_msgs::ImageConstPtr r_image_msg_;
	sensor_msgs::CameraInfoConstPtr l_info_msg_;
	sensor_msgs::CameraInfoConstPtr r_info_msg_;


	// ROS odometry subscribe
	message_filters::Subscriber<geometry_msgs::PoseWithCovarianceStamped> pose_sub_;

	rosbag::Bag bag;
	ros::Publisher pose_pub_;

	bool isComputing = false;

	void dataCb(const sensor_msgs::ImageConstPtr& l_image_msg,
							const sensor_msgs::ImageConstPtr& r_image_msg,
							const sensor_msgs::CameraInfoConstPtr& l_info_msg,
							const sensor_msgs::CameraInfoConstPtr& r_info_msg
						)
	{
		if(isComputing) // do not overide the pointer when computation is still going on
			return;
		
		all_received_++;
		l_image_msg_ = l_image_msg;
		r_image_msg_ = r_image_msg;
		l_info_msg_ = l_info_msg;
		r_info_msg_ = r_info_msg;
		// std::cout << "Received No." << all_received_ << "ros image" << std::endl;
	}

	static void increment(int* count)
	{
		// std::cout << "Pose Received: " << (*count) << std::endl;
		(*count)++;
	}

private:
	bool visualisation_on;
	DUtilsCV::Drawing::Plot implot;
	void resetPlot(){
		implot.create(480, 640, // x to the right, z to the bottom, top-left origin
		-13, // x-min
		2, // x-max
		- (13), // z-min
		- (-2) ); // z-max
	}

	void publishTrajectory(const cv::Mat &image, const std_msgs::Header &header){

		cv_bridge::CvImage cv_image = cv_bridge::CvImage(header, \
                sensor_msgs::image_encodings::BGR8, image);

		_trajectory_pub.publish(cv_image.toImageMsg());
	}
	
};

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TFeature>
demoDetector<TVocabulary, TDetector, TFeature>::demoDetector
	(const std::string &vocfile, int width, int height)
	: m_vocfile(vocfile),
		m_width(width), 
		m_height(height), exact_sync_(ExactPolicy(3), left_sub_, right_sub_, left_info_sub_, right_info_sub_), 
		all_received_(0), pose_received_(0)
{
	// Resolve topic names
	ros::NodeHandle nh;
	ros::NodeHandle local_nh("~");
  	// Subscribe to reset topic
    _reset_sub = nh.subscribe("/reset", 1, &demoDetector::resetCallback, this);

	_trajectory_pub = local_nh.advertise<sensor_msgs::Image>("trajectory",3);

	// load params
	ROS_ASSERT(local_nh.getParam("visualisation", visualisation_on));

	bag.open("/tmp/loopdetector_pose.bag", rosbag::bagmode::Write);
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TFeature>
void demoDetector<TVocabulary, TDetector, TFeature>::run
	(const std::string &name, const FeatureExtractor<TDescriptor> &extractor)
{
	std::cout << "======= DLoopDetector Demo - Modified by Cheng Huimin ===========" << std::endl;
	
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
	std::cout << "Loading " << name << " vocabulary..." << std::endl;
	TVocabulary voc(m_vocfile);
	
	// Initiate loop detector with the vocabulary 
	std::cout << "Processing sequence..." << std::endl;
	TDetector detector(voc, params); // typedef TemplatedLoopDetector
	
	// Process images
	std::vector<cv::KeyPoint> keys, keys_right;
	std::vector<TDescriptor> descriptors, descriptors_right;

	// load image filenames  
	// vector<string> filenames = 
	//   DUtils::FileFunctions::Dir(m_imagedir.c_str(), ".png", true);
	
	// load robot poses
	// vector<double> xs, ys;
	// readPoseFile(m_posefile.c_str(), xs, ys);
	
	// we can allocate memory for the expected number of images
	std::cout << "Allocating memory for detector" << std::endl;
	// detector.allocate(filenames.size());
	detector.allocate(1000);

	// prepare visualization windows
	// DUtilsCV::GUI::tWinHandler win = "Current image";
	DUtilsCV::GUI::tWinHandler winplot = "Trajectory";

	
	DUtilsCV::Drawing::Plot::Style normal_style('k',2, cv::LINE_AA); // thickness
	DUtilsCV::Drawing::Plot::Style loop_style('r', 2, cv::LINE_AA); // color, thickness
	// DUtilsCV::Drawing::Plot::Style detector_style('g', 2); // color, thickness

	resetPlot();
	
	// prepare profiler to measure times
	DUtils::Profiler profiler;


	// ROS INITIALISATION

	// Read local parameters
	ros::NodeHandle local_nh("~");
	// Resolve topic names
	ros::NodeHandle nh;

	std::string left_topic, right_topic;
	std::string left_info_topic, right_info_topic;
	std::string pose_in_topic;
	local_nh.param<std::string>("left_image",left_topic, "/stereo/left/image_rect_raw" );
	local_nh.param<std::string>("right_image",right_topic, "/stereo/right/image_rect_raw" );
	local_nh.param<std::string>("left_camerainfo",left_info_topic, "/stereo/left/camera_info" );
	local_nh.param<std::string>("right_camerainfo",right_info_topic, "/stereo/right/camera_info" );

	local_nh.param<std::string>("pose_input",pose_in_topic, "/stereo_odometer/pose" );

	// Subscribe to four input topics.
	ROS_INFO("dloopdetector: Subscribing to:\n\t* %s\n\t* %s\n\t* %s\n\t* %s\n\t* %s", 
				left_topic.c_str(), right_topic.c_str(),
				left_info_topic.c_str(), right_info_topic.c_str(), pose_in_topic.c_str());

	image_transport::ImageTransport it(nh);
	image_transport::TransportHints hints("raw",ros::TransportHints().tcpNoDelay());
	left_sub_.subscribe(it, left_topic, 1, hints); // http://docs.ros.org/diamondback/api/image_transport/html/classimage__transport_1_1TransportHints.html
	right_sub_.subscribe(it, right_topic, 1, hints);
	left_info_sub_.subscribe(nh, left_info_topic, 1,  ros::TransportHints().tcpNoDelay());
	right_info_sub_.subscribe(nh, right_info_topic, 1,  ros::TransportHints().tcpNoDelay());

	exact_sync_.registerCallback(boost::bind(&demoDetector::dataCb, this, _1, _2, _3, _4));

	pose_sub_.subscribe(nh, pose_in_topic,50);
	pose_sub_.registerCallback(boost::bind(&demoDetector::increment, &pose_received_));


	message_filters::Cache<geometry_msgs::PoseWithCovarianceStamped> pose_cache(pose_sub_, 100);

	pose_pub_ = local_nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose", 3);


	
	
	int count = 0;
	int db_size = 0;

	
	QuadMatcher<TDescriptor, TFeature> qm;
	StereoMotionEstimator sme;

	std::vector < geometry_msgs::PoseWithCovarianceStamped > m_pose_msg;
	m_pose_msg.reserve(500);

	// ros::AsyncSpinner spinner(2);
	// spinner.start();
	if (visualisation_on)
		DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 10); 

	ROS_INFO("Waiting for pose message from VO...");
	while (ros::ok() && !pose_received_)
	{
		ros::spinOnce();
		ros::Duration(0.5).sleep();
	}
	std::cout << "ROS Go!" << std::endl;
	// go
	while(true)
	{
		// get image from ROS
		while (ros::ok() && all_received_/10 <= db_size)
		{
			ros::spinOnce();
			ros::Duration(0.05).sleep();
		}

		if(!ros::ok())
		{
			std::cout << "ROS Shutting down" << std::endl;
			break;
		}

		isComputing = true;
		ros::Duration(0.5).sleep();
		ros::spinOnce();
		std::cout << "Adding image " << db_size << std::endl;

		// Retrive images from ROS
		auto cvImage_l = cv_bridge::toCvShare(l_image_msg_, sensor_msgs::image_encodings::MONO8);
		auto cvImage_r = cv_bridge::toCvShare(r_image_msg_, sensor_msgs::image_encodings::MONO8);

		ROS_ASSERT( m_width == (int)l_image_msg_->width && m_height == (int)l_image_msg_->height);
		cv::Mat im = cvImage_l->image;
		cv::Mat im_right = cvImage_r->image;

		uint64_t timestamp = cvImage_l->header.stamp.toNSec();
		

		// std::cout << "oldest: " << pose_cache.getOldestTime() << "latest: " << pose_cache.getLatestTime() << std::endl;
		// std::cout << "header: " << l_image_msg_->header.stamp << std::endl;

		auto pose_vec = pose_cache.getInterval(l_image_msg_->header.stamp,l_image_msg_->header.stamp); // std::vector< geometry_msgs::PoseWithCovarianceStampedConstPtr >

		if (pose_vec.size() == 1){
			
			m_pose_msg.push_back(*pose_vec[0]);
		}else
			std::cerr << "warning: pose_vec.size()=" << pose_vec.size() << std::endl;

		assert(pose_vec.size() <= 1);

		if (pose_vec.empty())
		{
			ros::Duration(0.5).sleep(); // give time for VO to process and publish
			ros::spinOnce();
			continue;
		}
			
		


		if (db_size == 0) // yet to initlaise
		{
			image_geometry::StereoCameraModel model;
			model.fromCameraInfo(*l_info_msg_, *r_info_msg_);

			StereoMotionEstimatorParam::Parameters sme_param;
			
			sme_param.ransac_iters = 400;
			sme_param.reweighting = false;
			sme_param.inlier_threshold = 4.0;
			sme_param.inlier_ratio_min = 0.3;
			sme_param.image_width = m_width;
			sme_param.image_height = m_height;

			sme_param.calib.baseline = model.baseline();
			sme_param.calib.f = model.left().fx();
			sme_param.calib.cu = model.left().cx();
			sme_param.calib.cv = model.left().cy();

			sme.setParam(sme_param);

			QuadMatcherParam::Parameters qm_param;
			qm_param.image_width = m_width;
			qm_param.image_height = m_height;
			qm_param.epipolar_tolerance = 10;
			qm_param.use_bucketing = true;
			qm_param.max_neighbor_ratio = 0.6;
			qm_param.compulte_scaled_keys =true;

			qm.setParam(qm_param);
		}
		

		//////////////////////////////////////////////////////////////////////////////
		
		// get features
		profiler.profile("features");
		extractor(im, keys, descriptors);
		extractor(im_right, keys_right, descriptors_right);
		profiler.stop();

		// show image
		// cv::Mat outIm(cv::Size(m_width,m_height),CV_8UC1);
		// cv::drawKeypoints(im,keys,outIm);
		// DUtilsCV::GUI::showImage(outIm, true, &win, 10);
				
		// add image to the collection and check if there is some loop
		typename TDetector::DetectionResult result;
		
		profiler.profile("detection");
		detector.detectLoop(keys, descriptors, result, keys_right, descriptors_right, timestamp); // db_size + 1 for each of the detectLoop operation
		db_size++;
		profiler.stop();
		
		if(result.detection())
		{
			std::cout << "- Loop found query image " << result.query << " with match image " << result.match << "!" << std::endl;

			profiler.profile("quadmatch");
			// l1, l2, r1, r2
			qm.pushBackData (result.match_keys.main, result.query_keys.main, 
											result.match_keys.right, result.query_keys.right, 
											result.match_descriptors.main, result.query_descriptors.main,
											result.match_descriptors.right, result.query_descriptors.right);
			bool match_result = qm.matchFeaturesQuad();
			profiler.stop();

			if (match_result)
			{
				profiler.profile("viso");
				// Obtain matches indices in vectors of 4-element arrays
				std::vector< std::array<int,4> > matches_quad_vec;
				qm.getMatchesQuad(matches_quad_vec);

				// previous left --> current left --> current right --> previous right
				sme.pushBackData(matches_quad_vec, result.match_keys.main, result.query_keys.main,
					result.query_keys.right, result.match_keys.right);

				bool viso_result = sme.updateMotion();
				profiler.stop();

				if (viso_result)
				{
					result.delta_pose_tr = sme.getCameraMotion();
					std::cout << "TR Estimate" << std::endl << sme.getCameraMotion().matrix() << std::endl;
				}	
				else
					std::cout << "viso: Error Updating Motion." << std::endl;

			}else
				std::cerr << "matchFeaturesQuad() Fails." << std::endl;

			++count;
		}
		else
		{
			std::cout << "- No loop: ";
			switch(result.status)
			{
				case CLOSE_MATCHES_ONLY:
					std::cout << "All the images in the database are very recent" << std::endl;
					break;
					
				case NO_DB_RESULTS:
					std::cout << "There are no matches against the database (few features in"
						" the image?)" << std::endl;
					break;
					
				case LOW_NSS_FACTOR:
					std::cout << "Little overlap between this image and the previous one"
						<< std::endl;
					break;
						
				case LOW_SCORES:
					std::cout << "No match reaches the score threshold (alpha: " <<
						params.alpha << ")" << std::endl;
					break;
					
				case NO_GROUPS:
					std::cout << "Not enough close matches to create groups. "
						<< "Best candidate: " << result.match << std::endl;
					break;
					
				case NO_TEMPORAL_CONSISTENCY:
					std::cout << "No temporal consistency (k: " << params.k << "). "
						<< "Best candidate: " << result.match << std::endl;
					break;
					
				case NO_GEOMETRICAL_CONSISTENCY:
					std::cout << "No geometrical consistency. Best candidate: " 
						<< result.match << std::endl;
					break;
					
				default:
					break;
			}
		}
		std::cout << " Loop detection (mean): " << profiler.getMeanTime("detection") * 1e3 << " ms/image" << std::endl;
		
		// obtain VO odometry pose, if detection found
		if(result.detection()){

			std::cout << "match_stamp=" << result.match_stamp;
			std::cout << ", query_stamp=" << result.query_stamp << std::endl;

			geometry_msgs::PoseWithCovarianceStamped match_pose_msg,query_pose_msg;
			size_t idx = 0;
			bool found = false;
			for(; idx < m_pose_msg.size() ; idx++)
			{
				if (m_pose_msg[idx].header.stamp.toNSec() == result.match_stamp)
				{
					match_pose_msg = m_pose_msg[idx];
					found = true;
					break;
				}
			}
			assert(found);
			found = false;
			for(; idx < m_pose_msg.size() ; idx++)
			{
				if (m_pose_msg[idx].header.stamp.toNSec() == result.query_stamp)
				{
					query_pose_msg = m_pose_msg[idx];
					found = true;
					break;
				}
			}
			assert(found);


			match_pose_msg.header.frame_id = "match_vo_frame";
			query_pose_msg.header.frame_id = "query_vo_frame";


			geometry_msgs::Pose detector_pose_msg = tf2::toMsg(sme.getCameraMotion());

			// Eigen::Affine3d match_pose_tr, query_pose_tr;
			tf2::fromMsg(match_pose_msg.pose.pose,result.match_pose_tr);
			tf2::fromMsg(query_pose_msg.pose.pose,result.query_pose_tr);

			auto delta_pose_tr =result.match_pose_tr.inverse() * result.query_pose_tr;
			geometry_msgs::Pose delta_pose = tf2::toMsg(delta_pose_tr);

			geometry_msgs::PoseWithCovarianceStamped pose_msg, delta_pose_msg;
			pose_msg.header.stamp.fromNSec(result.query_stamp);
			pose_msg.header.frame_id = "detector_frame";
			pose_msg.pose.pose = detector_pose_msg;

			delta_pose_msg.header.stamp = query_pose_msg.header.stamp;
			delta_pose_msg.header.frame_id = "delta_vo_frame";
			delta_pose_msg.pose.pose = delta_pose;

			bag.write("detector_msg",ros::Time::now(),query_pose_msg);
			bag.write("detector_msg",ros::Time::now(),match_pose_msg);
			bag.write("detector_msg",ros::Time::now(),delta_pose_msg);
			bag.write("detector_msg",ros::Time::now(),pose_msg);

			pose_pub_.publish(query_pose_msg);
			pose_pub_.publish(match_pose_msg);
			pose_pub_.publish(delta_pose_msg);
			pose_pub_.publish(pose_msg);
		}
		// // show trajectory
		if(m_pose_msg.size() > 1)
		{
			auto i = m_pose_msg.size() - 1;
			auto x1 = m_pose_msg[i-1].pose.pose.position.x;
			auto x2 = m_pose_msg[i].pose.pose.position.x;
			auto z1 = m_pose_msg[i-1].pose.pose.position.z;
			auto z2 = m_pose_msg[i].pose.pose.position.z;

			std::cout << "x1=" << x1 << ", x2=" << x2 << ", z1=" << z1 << ", z2=" << z2 << std::endl;

			cv::Mat image = implot.getImage();

			if(result.detection())
			{
				implot.line(x1, -z1, x2, -z2, loop_style);
				if (!result.delta_pose_tr.matrix().hasNaN()) // sanity check
				{
					auto loop_correct_tr = result.match_pose_tr * result.delta_pose_tr;
					double x = loop_correct_tr.translation()[0];
					double z = loop_correct_tr.translation()[2];
					double PxX = implot.toPxX(x);
					double PxY = implot.toPxY(-z);

					cv::drawMarker(image,cv::Point(PxX,PxY),cv::Scalar(0,200,0),cv::MARKER_STAR, 10, 1, cv::LINE_AA);
				}
			}
			else
				implot.line(x1, -z1, x2, -z2, normal_style);
			
			if (visualisation_on)
				DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 10); 

			std_msgs::Header header;
			header.stamp.fromNSec(result.query_stamp);
			header.frame_id = "dloopdetector";
			publishTrajectory(implot.getImage(),header);
		}

		isComputing = false;
	}
	
	if(count == 0)
	{
		std::cout << "No loops found in this image sequence" << std::endl;
	}
	else
	{
		std::cout << count << " loops found in this image sequence!" << std::endl;
	} 

	std::cout << std::endl << "Execution time:" << std::endl
		<< " - Feature computation : (mean) " << profiler.getMeanTime("features") * 1e3
		<< " ms/image, (max)" << profiler.getMaxTime("features") * 1e3
		<< " ms/image" << std::endl
		<< " - Loop detection (mean): " << profiler.getMeanTime("detection") * 1e3
		<< " ms/image, (max)" << profiler.getMaxTime("detection") * 1e3
		<< " ms/image" << std::endl;

		std::cout << "- Quad matching (mean): " << profiler.getMeanTime("quadmatch") * 1e3 << " ms/image" << std::endl;
		std::cout << "- Viso RT Estimation (mean): " << profiler.getMeanTime("viso") * 1e3 << " ms/image" << std::endl;

	std::cout << std::endl << "Press a key to finish..." << std::endl;
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
	
	std::fstream f(filename, std::ios::in);
	
	std::string s;
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

