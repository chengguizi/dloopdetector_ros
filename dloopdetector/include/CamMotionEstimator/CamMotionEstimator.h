/*
 * File: CamMotionEstimator.h
 * Date: September 2018
 * Author: Cheng Huimin
 * License: 
 * 
 * Estimate R and T from 4 sets of unmatched features.
 */

#ifndef CAM_MOTION_ESTIMATOR_H
#define CAM_MOTION_ESTIMATOR_H
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <vector>
#include <list>
#include <array>

#include <iostream>
#include <cassert>

// Class TFeature should implement function: double distance(const TDescriptor &a, const TDescriptor &b)
template<class TDescriptor, class TFeature>
class CamMotionEstimator{

public:
    typedef std::array<int,4> DMatch;

    CamMotionEstimator(int image_width, int image_height, int n_bucket_width = 8, int n_bucket_height = 8, 
        double max_neighbor_ratio = 0.6, bool use_bucketing = true) 
    :   image_width(image_width), image_height(image_height),
        data_ready(false), n_bucket_width(n_bucket_width), n_bucket_height(n_bucket_height), 
        max_neighbor_ratio(max_neighbor_ratio), use_bucketing(use_bucketing),
        // calculating the suitable bucket size for the given 2D bucket dimensions
        bucket_height( (image_height + n_bucket_height - 1 ) / n_bucket_height ), 
        bucket_width ( (image_width + n_bucket_width - 1 ) / n_bucket_width )
    {
        std::cout << "Creating CamMotionEstimator with image size " << image_width << "x" << image_height << ", bucketing " 
            << n_bucket_width << "x" << n_bucket_height << " ["<< bucket_width << "," << bucket_height << "]" << std::endl;
        // Reserve # of buckets space in the bucket vector
        bucketl1.resize(n_bucket_width*n_bucket_height);
        bucketr1.resize(n_bucket_width*n_bucket_height);
        bucketl2.resize(n_bucket_width*n_bucket_height);
        bucketr2.resize(n_bucket_width*n_bucket_height);
        
    };

    void pushBackData(const std::vector<cv::KeyPoint> &keyl1, const std::vector<cv::KeyPoint> &keyl2, 
                            const std::vector<cv::KeyPoint> &keyr1, const std::vector<cv::KeyPoint> &keyr2,
                            const std::vector<TDescriptor> &desl1, const std::vector<TDescriptor> &desl2,
                            const std::vector<TDescriptor> &desr1, const std::vector<TDescriptor> &desr2 );
    
    // Circular matching of 4 images
    // 1. previous left --> current left
    // 2. current left --> current right
    // 3. current right --> previous right
    // 4. previous right --> previous left

    bool matchFeaturesQuad(int epipolar_tolarance = 10 );

    void getMatchesQuad( std::vector< DMatch > &matches_quad );
private:

    enum HorizontalConstraint {NONE, LEFT_ONLY, RIGHT_ONLY};
    // Signal indicating that the quad image sequences are loaded, and
    // ready to be processed for matching and RT estimation.
    bool data_ready;

    // The width and height of the image sequences
    const int image_width, image_height;
    
    // The number of rows and columns that each image will be bucketed into.
    // Generally the height of the each bucket should be smaller, to better imposed the epipolar constraints
    bool use_bucketing;
    int epipolar_tolarance;
    const int n_bucket_width, n_bucket_height;
    const int bucket_height, bucket_width;

    // Maintain a vector list for each of the four image sequences, for indexing points in each bucket block
    std::vector< std::vector<int> > bucketl1, bucketr1, bucketl2, bucketr2;


    // Bucket index is ROW MAJOR
    inline int getBucketIndex(float x, float y);
    inline std::vector<int> getEpipolarBucketPoints(const cv::KeyPoint &key, 
                                                    const std::vector< std::vector<int> > &bucket,
                                                    const HorizontalConstraint constraint);
    void createBucketIndices(const std::vector<cv::KeyPoint> *keys , 
                                std::vector< std::vector<int> > &bucket);

    // Only store pointers to the actual data of keypoints and their descriptors to avoid copy overhead
    // NOTE: This class does not store data, so make sure the data is in scope throughout each processing iteration 
    const std::vector<cv::KeyPoint> *keyl1, *keyl2, *keyr1, *keyr2;
    const std::vector<TDescriptor> *desl1, *desl2, *desr1, *desr2;

    double max_neighbor_ratio;
    inline int findMatch(   const TDescriptor &query, const std::vector<TDescriptor> *targets );
    inline int findMatch(   const std::vector<int> &inside_bucket, const TDescriptor &query,
                            const std::vector<TDescriptor> *targets );

    
    void updateMatchList( const int matches_source, const int matches_target,
                    const std::vector<cv::KeyPoint> *key_sources, 
                    const std::vector< std::vector<int> > &bucket,
                    const std::vector<cv::KeyPoint> *key_targets,
                    const std::vector<TDescriptor> *des_sources, 
                    const std::vector<TDescriptor> *des_targets, 
                    const HorizontalConstraint constraint );
    // Matching results
    
    std::list< std::array<int,5> > matches;

};




///////////////////////////////////////////////////////////////////////
//// Implementation of Public Member Functions
///////////////////////////////////////////////////////////////////////


template<class TDescriptor, class TFeature>
void CamMotionEstimator<TDescriptor, TFeature>::pushBackData(
                            const std::vector<cv::KeyPoint> &keyl1, 
                            const std::vector<cv::KeyPoint> &keyl2, 
                            const std::vector<cv::KeyPoint> &keyr1, 
                            const std::vector<cv::KeyPoint> &keyr2, 
                            const std::vector<TDescriptor> &desl1, 
                            const std::vector<TDescriptor> &desl2,
                            const std::vector<TDescriptor> &desr1, 
                            const std::vector<TDescriptor> &desr2 ) {
    // Assignment to pointers
    this->keyl1 = &keyl1;
    this->keyr1 = &keyr1;
    this->keyl2 = &keyl2;
    this->keyr2 = &keyr2;

    this->desl1 = &desl1;
    this->desr1 = &desr1;
    this->desl2 = &desl2;
    this->desr2 = &desr2;

    assert( keyl1.size() == desl1.size() );
    assert( keyr1.size() == desr1.size() );
    assert( keyl2.size() == desl2.size() );
    assert( keyr2.size() == desr2.size() );

    std::cout << "Creating Bucket Indices" << std::endl;

    createBucketIndices(this->keyl1,bucketl1);
    createBucketIndices(this->keyr1,bucketr1);
    createBucketIndices(this->keyl2,bucketl2);
    createBucketIndices(this->keyr2,bucketr2);

    data_ready = true;
}



template<class TDescriptor, class TFeature>
bool CamMotionEstimator<TDescriptor, TFeature>::matchFeaturesQuad( int epipolar_tolarance ) {
    if (!data_ready)
    {
        std::cerr << "CamMotionEstimator ERROR: Data not data_ready." << std::endl;
        return false;
    }

    this->epipolar_tolarance = epipolar_tolarance;

    // cv::BFMatcher matcher;

    // Initialise the matches list with all points from 
    std::cout << "query # keys: " << keyl1->size() << ", match # keys: " << keyl2->size() << std::endl;

    matches.clear();

    updateMatchList (0, 1, keyl1, bucketl2, keyl2, desl1, desl2, NONE); // previous left --> current left
    updateMatchList (1, 2, keyl2, bucketr2, keyr2, desl2, desr2, LEFT_ONLY); // current left --> current right
    updateMatchList (2, 3, keyr2, bucketr1, keyr1, desr2, desr1, NONE); // current right --> previous right
    updateMatchList (3, 4, keyr1, bucketl1, keyl1, desr1, desl1, RIGHT_ONLY); // previous right --> previous left

    int list_size = matches.size();

    // Clean up wrong looped matches
    auto it = matches.begin();
    while (it != matches.end())
    {
        if ( (*it)[0] != (*it)[4]) // matching loop failed
        {
            it = matches.erase(it);
        }else
            it++;
    }
    std::cout << "Matching loop closes: " << matches.size() << " of " << list_size << std::endl;
    
    data_ready = false;
    return true;
}



template<class TDescriptor, class TFeature>
void CamMotionEstimator<TDescriptor, TFeature>::getMatchesQuad( std::vector< DMatch > &matches_quad ) {
    
    assert (matches_quad.empty());

    matches_quad.reserve(matches.size());

    for ( auto match : matches )
    {
        matches_quad.push_back( DMatch{match[0], match[1], match[2], match[3]} );
    }

}



///////////////////////////////////////////////////////////////////////
//// Implementation of Private Member Functions
///////////////////////////////////////////////////////////////////////



// OpenCV use top-left as origin, x as column/width, y as row/height
template<class TDescriptor, class TFeature>
int CamMotionEstimator<TDescriptor, TFeature>::getBucketIndex(float x, float y) {
    
    const int idx_y = y / bucket_height;
    const int idx_x = x / bucket_width;

    if ( !(idx_y >= 0 && idx_y < n_bucket_height && idx_x >= 0 && idx_x < n_bucket_width) )
    {
        std::cout << "x = " << x <<", idx_x = " << idx_x << ", y = " << y <<", idx_y = " << idx_y << std::endl;
        exit(-1);
    }

    int idx = idx_y*n_bucket_width + idx_x;
    assert (idx >= 0 && idx < n_bucket_height*n_bucket_width);

    return idx;
}

template<class TDescriptor, class TFeature>
std::vector<int> CamMotionEstimator<TDescriptor, TFeature>::getEpipolarBucketPoints(
        const cv::KeyPoint &key, const std::vector< std::vector<int> > &bucket, const HorizontalConstraint constraint)
{
    // Assume ROW MAJOR bucketing

    const float lower_bound_y = max( 0.f  , key.pt.y - epipolar_tolarance) ; // top-left origin x == width direction, https://stackoverflow.com/questions/25642532/opencv-pointx-y-represent-column-row-or-row-column
    const float upper_bound_y = key.pt.y + epipolar_tolarance;

    const float lower_bound_x = ( constraint == RIGHT_ONLY ? key.pt.x : 0);
    const float upper_bound_x = ( constraint == LEFT_ONLY ? key.pt.x : image_width-1 );

    const int lower_bound_bucket = getBucketIndex( lower_bound_x, lower_bound_y);
    const int upper_bound_bucket = getBucketIndex( upper_bound_x , upper_bound_y);

    std::vector<int> results;

    for (int i = lower_bound_bucket ; i <= upper_bound_bucket; i++)
    {
        for ( auto e : bucket[i] )
        {
            results.push_back(e);
        }
    }

    return results;
}

template<class TDescriptor, class TFeature>
void CamMotionEstimator<TDescriptor, TFeature>::createBucketIndices(const std::vector<cv::KeyPoint> *keys , 
                                                    std::vector< std::vector<int> > &bucket) {
    // Clear the previous bucket cache
    for ( auto &element : bucket )
        element.clear();

    // Iterate all keypoints and add them to the correct bucket list
    for (int i = 0 ; i < keys->size() ; i++)
    {
        const cv::KeyPoint &key = (*keys)[i];
        int idx = getBucketIndex(key.pt.x, key.pt.y);
        bucket[idx].push_back(i);
    }
    
}

template<class TDescriptor, class TFeature>
int CamMotionEstimator<TDescriptor, TFeature>::findMatch(const TDescriptor &query,
                            const std::vector<TDescriptor> *targets )
{
    double best_dist_1 = 1e9;
    double best_dist_2 = 1e9;
    int best_i = -1;

    //// Not using bucketing
    for (int i = 0; i < targets->size(); i++)
    {
        double dist = TFeature::distance(query,(*targets)[i]);

        if (dist < best_dist_1 ) {
            best_i = i;
            best_dist_2 = best_dist_1;
            best_dist_1 = dist;
        }else if (dist < best_dist_2) {
            best_dist_2 = dist;
        }
    }

    // If the best match is much better than the second best, then it is most probably a good match (SNR good)
    if ( best_dist_1 / best_dist_2 < max_neighbor_ratio)
    {
        return best_i;
    }
    return -1;
}

template<class TDescriptor, class TFeature>
int CamMotionEstimator<TDescriptor, TFeature>::findMatch( const std::vector<int> &inside_bucket, const TDescriptor &query,
                            const std::vector<TDescriptor> *targets )
{
    double best_dist_1 = 1e9;
    double best_dist_2 = 1e9;
    int best_i = -1;

    // there is no suitable bucketed target points to be matched, early return
    if (inside_bucket.empty())
        return -1;

    // std::cout << "Bucketing " << inside_bucket.size() << " target features out of " << targets->size() << std::endl;

    for ( int idx = 0 ; idx < inside_bucket.size() ; idx++ )
    {
        int i = inside_bucket[idx];

        double dist = TFeature::distance(query,(*targets)[i]);

        if (dist < best_dist_1 ) {
            best_i = i;
            best_dist_2 = best_dist_1;
            best_dist_1 = dist;
        }else if (dist < best_dist_2) {
            best_dist_2 = dist;
        }
    }

    // If the best match is much better than the second best, then it is most probably a good match (SNR good)
    if ( best_dist_1 / best_dist_2 < max_neighbor_ratio)
    {
        return best_i;
    }
    return -1;
}

template<class TDescriptor, class TFeature>
void CamMotionEstimator<TDescriptor, TFeature>::updateMatchList( 
                const int matches_source, 
                const int matches_target,
                const std::vector<cv::KeyPoint> *key_sources, 
                const std::vector< std::vector<int> > &bucket,
                const std::vector<cv::KeyPoint> *key_targets,
                const std::vector<TDescriptor> *des_sources, 
                const std::vector<TDescriptor> *des_targets,
                const HorizontalConstraint constraint  ) {

    // if matches_source is zero, means the match list needs initialisation
    if (matches_source == 0) {

        assert (matches.empty());

        for ( int i = 0; i < (*key_sources).size() ; i++ )
        {
            const TDescriptor &des = (*des_sources)[i];
            const cv::KeyPoint &key = (*key_sources)[i];
            int j;
            if (!use_bucketing) 
                j = findMatch( des, des_targets );
            else
                j = findMatch( getEpipolarBucketPoints(key, bucket, constraint), des, des_targets );

            if (j >= 0)
            {
                matches.push_back({i,j,0,0,0});
            }
        }

        std::cout << "Found " << matches.size() << " matches between 0 and 1" << std::endl;
    }
    // else, use the [matches_source] column as the query points
    else {

        int list_size = matches.size();

        auto it = matches.begin();
        while(it != matches.end())
        {
            int i = (*it)[matches_source];

            const TDescriptor &des = (*des_sources)[i];
            const cv::KeyPoint &key = (*key_sources)[i];
            int j;
            if (!use_bucketing) 
                j = findMatch( des, des_targets );
            else
                j = findMatch( getEpipolarBucketPoints(key, bucket, constraint), des, des_targets );

            // Found matches for ith keypoint in the source image to the jth keypoint in the target image
            if (j >= 0)
            {
                (*it)[matches_target] = j;
                it++;
            }
            // failure to find, delete this node, and proceed to the next node
            else
            {
                it = matches.erase(it);
            }
        }

        std::cout << "Matches " << matches.size() << " out of previous " << list_size << " from " 
            <<  matches_source << " to " << matches_target << std::endl;
    }
}

#endif // CAM_MOTION_ESTIMATOR_H