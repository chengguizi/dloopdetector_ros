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

#define UNUSED(x) (void)(x)

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <vector>
#include <list>
#include <array>

#include <iostream>
#include <cassert>

#include <bitset>
#include <string>

int distance_counter = 0;

// Class TFeature should implement function: double distance(const TDescriptor &a, const TDescriptor &b)
template<class TDescriptor, class TFeature>
class CamMotionEstimator{

public:
    typedef std::array<int,4> DMatch;

    struct Parameters {
        // The width and height of the image sequences
        int image_width, image_height;

        // The number of rows and columns that each image will be bucketed into.
        // Generally the height of the each bucket should be smaller, to better imposed the epipolar constraints
        int n_bucket_width = 8, n_bucket_height = 8;
        int epipolar_tolarance = 10;
        double max_neighbor_ratio = 0.6;
        bool use_bucketing= true;

        // This will be set automatically, do not change
        int bucket_height;
        int bucket_width;
    };

    CamMotionEstimator() : initialised(false), data_ready(false) {
        int num_bucket = param.n_bucket_width * param.n_bucket_height;
        // Reserve # of buckets space in the bucket vector
        bucketl1.resize(num_bucket);
        bucketr1.resize(num_bucket);
        bucketl2.resize(num_bucket);
        bucketr2.resize(num_bucket);
    };

    void setParam(Parameters &param){

        // calculating the suitable bucket size for the given 2D bucket dimensions
        param.bucket_height = (param.image_height + param.n_bucket_height - 1 ) / param.n_bucket_height;
        param.bucket_width = (param.image_width + param.n_bucket_width - 1 ) / param.n_bucket_width;

        std::cout << std::endl << "CamMotionEstimator parameters loaded!" << std::endl;
        std::cout << "- image size " << param.image_width << "x" << param.image_height << std::endl
            << "- bucketing " << param.n_bucket_width << "x" << param.n_bucket_height 
            << " ["<< param.bucket_width << "," << param.bucket_height << "]" << std::endl;

        std::cout << "==============================================================" << std::endl;
        
        this->param = param;
        initialised = true;
    }

    void pushBackData(const std::vector<cv::KeyPoint> &keyl1, const std::vector<cv::KeyPoint> &keyl2, 
                            const std::vector<cv::KeyPoint> &keyr1, const std::vector<cv::KeyPoint> &keyr2,
                            const std::vector<TDescriptor> &desl1, const std::vector<TDescriptor> &desl2,
                            const std::vector<TDescriptor> &desr1, const std::vector<TDescriptor> &desr2 );
    
    // Circular matching of 4 images
    // 1. previous left --> current left
    // 2. current left --> current right
    // 3. current right --> previous right
    // 4. previous right --> previous left

    bool matchFeaturesQuad();

    void getMatchesQuad( std::vector< DMatch > &matches_quad );
private:

    bool initialised;

    // Indicate whether to use epipolar constraints (if enabled) to bucket points before matching (so less points to match)
    enum HorizontalConstraint {NONE, LEFT_ONLY, RIGHT_ONLY};

    // Signal indicating that the quad image sequences are loaded, and
    // ready to be processed for matching and RT estimation.
    bool data_ready;

    // Containing all the tunable configurations
    Parameters param;

    // Maintain a vector list for each of the four image sequences, for indexing points in each bucket block
    std::vector< std::vector<int> > bucketl1, bucketr1, bucketl2, bucketr2;

    // Bucket index is ROW MAJOR
    inline int getBucketIndex(float x, float y);
    inline std::vector<int> getEpipolarBucketPoints(const cv::KeyPoint &key, 
                                                    const std::vector< std::vector<int> > &bucket,
                                                    const HorizontalConstraint constraint);
    void createBucketIndices(const std::vector<cv::KeyPoint> *keys , 
                                std::vector< std::vector<int> > &bucket);

    

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
    
    
    // Only store pointers to the actual data of keypoints and their descriptors to avoid copy overhead
    // NOTE: This class does not store data, so make sure the data is in scope throughout each processing iteration 
    const std::vector<cv::KeyPoint> *keyl1, *keyl2, *keyr1, *keyr2;
    const std::vector<TDescriptor> *desl1, *desl2, *desr1, *desr2;

    // Matching results
    std::list< std::array<int,5> > matches;
};




///////////////////////////////////////////////////////////////////////
//// Implementation of Public Member Functions
///////////////////////////////////////////////////////////////////////

// l1 - previous left (match), l2 current left (query)
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

    if (!initialised)
    {
        std::cerr << "CamMotionEstimator NOT initialised." << std::endl;
        return;
    }
        
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
bool CamMotionEstimator<TDescriptor, TFeature>::matchFeaturesQuad() {
    if (!data_ready)
    {
        std::cerr << "CamMotionEstimator ERROR: Data not data_ready." << std::endl;
        return false;
    }

    // cv::BFMatcher matcher;

    // Initialise the matches list with all points from 
    std::cout << "main: query # keys: " << keyl2->size() << ", match # keys: " << keyl1->size() << std::endl;
    std::cout << "right: query # keys: " << keyr2->size() << ", match # keys: " << keyl1->size() << std::endl;

    matches.clear();

    int pre = distance_counter;
    // TODO: for the NONE case, the homographic transformation could be estimated after a few initial match
    updateMatchList (0, 1, keyl1, bucketl2, keyl2, desl1, desl2, NONE); // previous left --> current left
    updateMatchList (1, 2, keyl2, bucketr2, keyr2, desl2, desr2, LEFT_ONLY); // current left --> current right
    updateMatchList (2, 3, keyr2, bucketr1, keyr1, desr2, desr1, NONE); // current right --> previous right
    updateMatchList (3, 4, keyr1, bucketl1, keyl1, desr1, desl1, RIGHT_ONLY); // previous right --> previous left

        

    std::cout << "distance() routine is called " << distance_counter - pre <<  " times" << std::endl;
    

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
    return matches.size();
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
    
    const int idx_y = y / param.bucket_height;
    const int idx_x = x / param.bucket_width;

    if ( !(idx_y >= 0 && idx_y < param.n_bucket_height && idx_x >= 0 && idx_x < param.n_bucket_width) )
    {
        std::cout << "x = " << x <<", idx_x = " << idx_x << ", y = " << y <<", idx_y = " << idx_y << std::endl;
        exit(-1);
    }

    int idx = idx_y*param.n_bucket_width + idx_x;
    assert (idx >= 0 && idx < param.n_bucket_height*param.n_bucket_width);

    return idx;
}

template<class TDescriptor, class TFeature>
std::vector<int> CamMotionEstimator<TDescriptor, TFeature>::getEpipolarBucketPoints(
        const cv::KeyPoint &key, const std::vector< std::vector<int> > &bucket, const HorizontalConstraint constraint)
{
    // Assume ROW MAJOR bucketing

    const float lower_bound_y = std::max( 0.f  , key.pt.y - param.epipolar_tolarance) ; // top-left origin x == width direction, https://stackoverflow.com/questions/25642532/opencv-pointx-y-represent-column-row-or-row-column
    const float upper_bound_y = std::min( param.image_height - 1.f, key.pt.y + param.epipolar_tolarance );

    const float lower_bound_x = ( constraint == RIGHT_ONLY ? key.pt.x : 0);
    const float upper_bound_x = ( constraint == LEFT_ONLY ? key.pt.x : param.image_width-1 );

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
    
    
    assert ( (int)bucket.size() == param.n_bucket_width * param.n_bucket_height);

    // Clear the previous bucket cache
    for ( auto &element : bucket )
        element.clear();

    // Iterate all keypoints and add them to the correct bucket list
    for (size_t i = 0 ; i < keys->size() ; i++)
    {
        const cv::KeyPoint &key = (*keys)[i];
        int idx = getBucketIndex(key.pt.x, key.pt.y);
        bucket[idx].push_back(i);
    }

    // Debug Output
    // for (int i = 0 ; i < bucket.size() ; i++)
    // {
    //     std::cout << "Bucket [" << i << "]:" << std::endl;
    //     for (int j = 0; j < bucket[i].size() ; j++)
    //     {
    //         cout << bucket[i][j] << ":(" << (*keys)[ bucket[i][j] ].pt.x << "," << (*keys)[ bucket[i][j] ].pt.y << ") ";
    //     }
    //     std::cout << std::endl;
    // }
    
}

template<class TDescriptor, class TFeature>
int CamMotionEstimator<TDescriptor, TFeature>::findMatch(const TDescriptor &query,
                            const std::vector<TDescriptor> *targets )
{
    int best_dist_1 = 1e9;
    int best_dist_2 = 1e9;
    int best_i = -1;

    //// Not using bucketing
    for (size_t i = 0; i < targets->size(); i++)
    {
        int dist = (query^(*targets)[i]).count(); //TFeature::distance(query,(*targets)[i]);
        distance_counter++;

        if (dist < best_dist_1 ) {
            best_i = i;
            best_dist_2 = best_dist_1;
            best_dist_1 = dist;
        }else if (dist < best_dist_2) {
            best_dist_2 = dist;
        }
    }

    // If the best match is much better than the second best, then it is most probably a good match (SNR good)
    if ( best_dist_1 / (double)best_dist_2 < param.max_neighbor_ratio)
    {
        return best_i;
    }
    return -1;
}

template<class TDescriptor, class TFeature>
int CamMotionEstimator<TDescriptor, TFeature>::findMatch( const std::vector<int> &inside_bucket, const TDescriptor &query,
                            const std::vector<TDescriptor> *targets )
{
    int best_dist_1 = 1e9;
    int best_dist_2 = 1e9;
    int best_i = -1;

    // there is no suitable bucketed target points to be matched, early return
    if (inside_bucket.empty())
        return -1;

    // std::cout << "Bucketing " << inside_bucket.size() << " target features out of " << targets->size() << std::endl;

    for ( size_t idx = 0 ; idx < inside_bucket.size() ; idx++ )
    {
        int i = inside_bucket[idx];

        int dist = (query^(*targets)[i]).count(); //TFeature::distance(query,(*targets)[i]);
        distance_counter++;

        if (dist < best_dist_1 ) {
            best_i = i;
            best_dist_2 = best_dist_1;
            best_dist_1 = dist;
        }else if (dist < best_dist_2) {
            best_dist_2 = dist;
        }
    }

    // If the best match is much better than the second best, then it is most probably a good match (SNR good)
    if ( best_dist_1 / (double)best_dist_2 < param.max_neighbor_ratio)
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

    UNUSED(key_targets);

    // if matches_source is zero, means the match list needs initialisation
    if (matches_source == 0) {

        assert (matches.empty());

        for ( size_t i = 0; i < (*key_sources).size() ; i++ )
        {
            const TDescriptor &des = (*des_sources)[i];
            const cv::KeyPoint &key = (*key_sources)[i];
            int j;
            if (param.use_bucketing && constraint != NONE) 
                j = findMatch( getEpipolarBucketPoints(key, bucket, constraint), des, des_targets );
            else
                j = findMatch( des, des_targets );

            if (j >= 0)
            {
                matches.push_back({(int)i,j,0,0,0});
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
            if (param.use_bucketing && constraint != NONE) 
                j = findMatch( getEpipolarBucketPoints(key, bucket, constraint), des, des_targets );
            else
                j = findMatch( des, des_targets );

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