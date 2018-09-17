/*
 * File: VisoStereo.h
 * Created: September 2018
 * Modified: September 2018
 * Author: Cheng Huimin
 * License: 
 * 
 * Description: A rewrite of original libviso2 into modern Eigen library. This is implemented as a header-only library itself
 */

#ifndef VISO_STEREO_H
#define VISO_STEREO_H

#include <Eigen/Eigen>
#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <vector>
#include <array>

#include <random>
#include <cmath>

#include <iostream>

class VisualOdometryStereo {

public:

    typedef std::array<int,4> DMatch;
    static const int L1 = 0, L2 = 1, R2 = 2, R1 = 3; // previous left, current left, current right, previous right

    // camera parameters (all are mandatory / need to be supplied)
    struct Calibration {  
        double f;  // focal length (in pixels)
        double cu; // principal point (u-coordinate) aka width
        double cv; // principal point (v-coordinate) aka height
        
        Calibration () {
        f  = 1;
        cu = 0;
        cv = 0;
        }
    };

    // stereo-specific parameters (mandatory: base)
	struct Parameters{
		double  base;             // baseline (meters)
		int32_t ransac_iters;     // number of RANSAC iterations
		double  inlier_threshold; // fundamental matrix inlier threshold
		bool    reweighting;      // lower border weights (more robust to calibration errors)
        int image_width;
        int image_height;
		
        Parameters() {
			base = 1.0;
			ransac_iters = 400;
			inlier_threshold = 2.0;
			reweighting = true;
		}

        Calibration calib;
	};

    // constructor, takes as inpute a parameter structure
    VisualOdometryStereo() : initialised(false) {}

    void setParam(Parameters param) { 
        this->param = param; 
        initialised = true;
        std::cout << "VisualOdometryStereo parameters loaded!" << std::endl;
        std::cout << "base=" << this->param.base << ", f=" << this->param.calib.f << ", cu=" << this->param.calib.cu << 
            ", cv=" << this->param.calib.cv << std::endl;
    }

    // Column sequence of matches_quad_vec: previous left, current left, current right, previous right
    void pushBackData(
        const std::vector<DMatch> &matches_quad_vec,
        const std::vector<cv::KeyPoint> &keyl1_vec,
        const std::vector<cv::KeyPoint> &keyl2_vec,
        const std::vector<cv::KeyPoint> &keyr2_vec,
        const std::vector<cv::KeyPoint> &keyr1_vec
    );

    
	bool updateMotion();

	std::vector<int> getInlier() { return getInlier(tr_delta_vec); };
	
	Eigen::Matrix<double,4,4> getMotion() { return Tr_delta; }

    // deconstructor
	~VisualOdometryStereo() {}

	friend std::ostream& operator<< (std::ostream &os,VisualOdometryStereo &viso) {
		auto p = viso.getMotion();
		os << p(0,0) << " " << p(0,1) << " "  << p(0,2)  << " "  << p(0,3) << " ";
		os << p(1,0) << " " << p(1,1) << " "  << p(1,2)  << " "  << p(1,3) << " ";
		os << p(2,0) << " " << p(2,1) << " "  << p(2,2)  << " "  << p(2,3);
		return os;
	}

private:
    
    
    enum                 result { UPDATED, FAILED, CONVERGED };

    void                 computeObservations(const std::vector<int> &active, bool inlier_mode = false);
    
    result               updateParameters(std::vector<int> &active, std::vector<double> &tr, double step_size, double eps);
	void                 computeResidualsAndJacobian(const std::vector<double> &tr, const std::vector<int> &active, bool inlier_mode = false);

	std::vector<double> estimateMotion();

	std::vector<int>    getInlier(std::vector<double> &tr);

    std::vector<int>    getRandomSample (int N,int num);

	Eigen::Matrix<double,4,4> transformationVectorToMatrix(const std::vector<double> &tr);

    bool initialised;
    Parameters param;

    const std::vector<DMatch> *matches_quad_vec;
    const std::vector<cv::KeyPoint> *keyl1_vec, *keyl2_vec, *keyr1_vec, *keyr2_vec;


    std::vector<double> X,Y,Z,J; // 3D points and Jacobian
    std::vector<double> p_observe;  // observed 2d points
    std::vector<double> p_predict;  // predicted 2d points
    std::vector<double> p_residual; // residuals (p_residual=p_observe-p_predict)

    std::vector<int> inliers;

	std::vector<double> tr_delta_vec;
	Eigen::Matrix<double,4,4> Tr_delta;
};




/////////////////////////////////////////////////////////////////////////////////
//// Implementation of Public Member Functions
/////////////////////////////////////////////////////////////////////////////////

void VisualOdometryStereo::pushBackData(
                                            const std::vector<DMatch> &matches_quad_vec,
                                            const std::vector<cv::KeyPoint> &keyl1_vec,
                                            const std::vector<cv::KeyPoint> &keyl2_vec,
                                            const std::vector<cv::KeyPoint> &keyr2_vec,
                                            const std::vector<cv::KeyPoint> &keyr1_vec
                                            ){
    this->matches_quad_vec = &matches_quad_vec;
    this->keyl1_vec = &keyl1_vec;
    this->keyl2_vec = &keyl2_vec;
    this->keyr1_vec = &keyr1_vec;
    this->keyr2_vec = &keyr2_vec;

	// Sanity checks
	// The matched index amount should be smaller / equal to available points
	assert(matches_quad_vec.size() <= min( keyr1_vec.size() ,  keyr2_vec.size() ));
	
}

bool VisualOdometryStereo::updateMotion () {
  
  // estimate motion
  vector<double> tr_delta = estimateMotion();
  tr_delta_vec.clear();
  
  // on failure
  if (tr_delta.size()!=6)
    return false;
  
  // set transformation matrix (previous to current frame)
  Tr_delta = transformationVectorToMatrix(tr_delta);
  tr_delta_vec = tr_delta;
  
  // success
  return true;
}


/////////////////////////////////////////////////////////////////////////////////
//// Implementation of Private Member Functions
/////////////////////////////////////////////////////////////////////////////////

vector<double> VisualOdometryStereo::estimateMotion()
{

	// compute minimum distance for RANSAC samples
	float width_max = 0, height_max = 0;
	float width_min = 1e5, height_min = 1e5;


    // with reference to previous left frame
	for ( const auto match : (*matches_quad_vec) )
	{
        const cv::KeyPoint key = (*keyl1_vec)[ match[L1] ];

		if ( key.pt.x > width_max)  width_max = key.pt.x;
		if ( key.pt.x < width_min)  width_min = key.pt.x;

		if ( key.pt.y > height_max) height_max = key.pt.y;
		if ( key.pt.y < height_min) height_min = key.pt.y;
	}

	// Defined the min-dist between any random 3 matches
	float min_dist = min(width_max-width_min, height_max-height_min) / 3.f;	// default divided by 3.0

    if ( min_dist < param.image_height / 10.f && min_dist < param.image_width / 10.f  )
    {
        std::cerr << "min_dist is too small (< 0.1*image_dimensions), aborting viso: " << min_dist << std::endl;
        return vector<double>();
    }

    min_dist = min_dist*min_dist;

    const int N = matches_quad_vec->size();
    if (N < 10)
    {
        std::cerr << "Total poll of matches is too small: " << N << std::endl;
        return vector<double>();
    }

	// clear vectors
	inliers.clear();
    X.resize(N);
    Y.resize(N);
    Z.resize(N);
    J.resize(4 * N * 6); // yx: save Jacobian matrix for each point (6*4: 6 functions and 4 unknowns)

    p_predict.resize(4 * N);
    p_observe.resize(4 * N);
    p_residual.resize(4 * N);

    double &_cu = param.calib.cu;
    double &_cv = param.calib.cv;
    double &_f = param.calib.f;

	// project matches of previous image into 3d

	for (int i = 0; i < (*matches_quad_vec).size() ; i++ )
	{
		const cv::KeyPoint keyl1 = (*keyl1_vec)[ (*matches_quad_vec)[i][L1] ];
		const cv::KeyPoint keyr1 = (*keyr1_vec)[ (*matches_quad_vec)[i][R1] ];

		if ( keyl1.pt.x - keyr1.pt.x < 0.0 )
		{
			std::cerr << "Warning: Flipped match at " << i << std::endl;
		}

		double d = max( keyl1.pt.x - keyr1.pt.x, 0.0001f);			// d = xl - xr
		X[i] = (keyl1.pt.x - _cu) * param.base / d;				// X = (u1p - calib.cu)*baseline/d
		Y[i] = (keyl1.pt.y - _cv) * param.base / d;				// Y = (v1p - calib.cv)*baseline/d
		Z[i] = _f * param.base / d;									// Z = f*baseline/d

	}



	// loop variables
	std::vector<double> tr_delta;				// yx: ransac: bestfit
	std::vector<double> tr_delta_curr(6,0);

	std::vector<int> best_active;

	// // initial RANSAC estimate
	for (int k = 0; k < param.ransac_iters; k++)
	{
        // std::cout << "Begin k=" << k << std::endl;
        // active stores the current active rows in matches_quad_vec
		std::vector<int> active; 


        // Generate 3 points that satisfied the min_dist, in the previous left image
		for (int selection_iter = 0; ; selection_iter++)
		{
			active = getRandomSample(N, 3);

            int idx0 = (*matches_quad_vec)[ active[0] ][L1];
            int idx1 = (*matches_quad_vec)[ active[1] ][L1];
            int idx2 = (*matches_quad_vec)[ active[2] ][L1];

            const cv::Point2f pt0 = (*keyl1_vec)[idx0].pt;
            const cv::Point2f pt1 = (*keyl1_vec)[idx1].pt;
            const cv::Point2f pt2 = (*keyl1_vec)[idx2].pt;

			double x0 = (pt0.x - pt1.x)*(pt0.x - pt1.x);
			double y0 = (pt0.y - pt1.y)*(pt0.y - pt1.y);
			double x1 = (pt1.x - pt2.x)*(pt1.x - pt2.x);
			double y1 = (pt1.y - pt2.y)*(pt1.y - pt2.y);
			double x2 = (pt2.x - pt0.x)*(pt2.x - pt0.x);
			double y2 = (pt2.y - pt0.y)*(pt2.y - pt0.y);
			double d0 = x0 + y0; double d1 = x1 + y1; double d2 = x2 + y2;

			if (d0 >= min_dist && d1 >= min_dist && d2 >= min_dist)
				break;

			if (selection_iter >= 100)
            {
                std::cerr << "Finding RANSAC 3 matches not possible..." << std::endl;
                return vector<double>();
            }
		}

        assert (active.size() == 3);
        // std::cout << "Random 3 Generated: " << active[0] << ", " << active[1] << ", " << active[2] << std::endl;
		
        // Initialise current TR estimation
        tr_delta_curr.clear();
        tr_delta_curr.resize(6,0);			// yx: ransac: maybemodel (for current sample)

		// minimize reprojection errors
        computeObservations(active);
		

        // std::cout << "Right Before updateParameters Routine" << std::endl;

        // std::cout << "p_observe: ";
        // for (int i=0;i<4;i++)
        //     std::cout << p_observe[i] << ' ';
        // std::cout << std::endl;

		VisualOdometryStereo::result result = UPDATED;
		for ( int iter = 1; result == UPDATED ; iter++ )
		{
			result = updateParameters(active, tr_delta_curr, 1, 1e-6);
			if (result == CONVERGED)
            {
                // std::cout << "RANSAC " << k << ": TR Converged after " << iter << " iterations. " << std::endl;
                break;
            }	

			if (iter >= 20) // hm: this happens very frequently
				break;
		}

        // std::cout << "Right After updateParameters Routine" << std::endl;

        // std::cout << "p_predict: ";
        // for (int i=0;i<4;i++)
        //     std::cout << p_predict[i] << ' ';
        // std::cout << std::endl;

        // std::cout << "p_residual: ";
        // for (int i=0;i<12;i++)
        //     std::cout << p_residual[i] << ' ';
        // std::cout << std::endl;

        // std::cout << " tr_delta_curr: " ;


        // if ( result == UPDATED)
        // {
        //     // std::cerr<< "RANSAC " << k <<  ": updateParameters() EXCEED 20 iterations." << std::endl;
        //     continue;
        // }

        if (result == FAILED)
            continue;

		// Update best inlier buffer if we have more inliers
        std::vector<int> inliers_curr = getInlier(tr_delta_curr);
        if (inliers_curr.size() > inliers.size())
        {
            inliers = inliers_curr;
            tr_delta = tr_delta_curr;
			best_active = active;
        }

        // std::cout << "inlier: " << inliers_curr.size() << " out of " << N << std::endl;

	// 	// probility of observing an inlier
	// 	/*
	// 	double Pin = double(inliers.size()) / double(N);
	// 	if (Pin > Pbest) Pbest = Pin;
	// 	N_ransac2 = log(1 - p) / log(1 - pow(Pbest, 3));

	// 	// N_ransac = 200 means, Pbest > 0.28
	// 	if (N_ransac2 < N_ransac &&  k < N_ransac2)
	// 	{
	// 		N_ransac = N_ransac2;
	// 	}
	// 	*/

    // std::cout << "End k=" << k << std::endl;
	}

    std::cout << "Best inlier: " << inliers.size() << " out of " << N << std::endl;

	// Sanity Check
	if ( inliers.size() / (double)N < 0.5 )
	{
		std::cout << "ERROR: Inlier % too small! Return false." << std::endl;
		return vector<double>();
	}

	std::cout << "Pre-Refinement TR vector: ";
	for (int i=0;i<6;i++)
		cout << tr_delta[i] << ", ";
	std::cout << std::endl;
	
	assert (tr_delta.size() == 6);
	// final optimization (refinement)
	int iter = 0;
	VisualOdometryStereo::result result = UPDATED;
	computeObservations(inliers);
	while (result == UPDATED)
	{
		result = updateParameters(inliers, tr_delta, 1, 1e-8);
		if (result == CONVERGED)
			break;
		if (iter++ > 100)
			break;
	}

	if (result == FAILED)
	{
		cerr << "WARNING refinement step -> updateParameters FAILED" << endl;
		return vector<double>();
	}

	// not converged
	if (result == UPDATED)
	{
		cerr << "WARNING refinement step -> updateParameters NOT converged" << endl;
		return vector<double>();
	}

	std::cout << "Post-Refinement TR vector: ";
	for (int i=0;i<6;i++)
		cout << tr_delta[i] << ", ";
	std::cout << std::endl;
	
	return tr_delta;
}

void VisualOdometryStereo::computeObservations(const std::vector<int> &active, bool inlier_mode ) {


    for (int i=0 ; i < active.size() || (inlier_mode && i < matches_quad_vec->size() ) ; i++)
    {
        int idx_l2 = (*matches_quad_vec)[ (inlier_mode ? i : active[i]) ][L2];
        int idx_r2 = (*matches_quad_vec)[ (inlier_mode ? i : active[i]) ][R2];
        p_observe[4*i+0] = (*keyl2_vec)[ idx_l2 ].pt.x;
        p_observe[4*i+1] = (*keyl2_vec)[ idx_l2 ].pt.y;
        p_observe[4*i+2] = (*keyr2_vec)[ idx_r2 ].pt.x;
        p_observe[4*i+3] = (*keyr2_vec)[ idx_r2 ].pt.y;
    }
//   // set all observations
//   for (int32_t i=0; i<(int32_t)active.size(); i++) {
//     p_observe[4*i+0] = p_matched[active[i]].u1c; // u1
//     p_observe[4*i+1] = p_matched[active[i]].v1c; // v1
//     p_observe[4*i+2] = p_matched[active[i]].u2c; // u2
//     p_observe[4*i+3] = p_matched[active[i]].v2c; // v2
//   }
}

VisualOdometryStereo::result VisualOdometryStereo::updateParameters(std::vector<int> &active, std::vector<double> &tr, double step_size, double eps) {

	// extract observations and compute predictions
	computeResidualsAndJacobian(tr, active);

	// init
    Eigen::Matrix<double,6,6> A;
    Eigen::Matrix<double,6,1> B;

	// fill matrices A and B
	// JT*J = A
	for (int m = 0; m < 6; m++)
	{
		for (int n = 0; n < 6; n++)
		{
			double a = 0;
			for (int i = 0; i < 4 * active.size(); i++)
			{
				a += J[i * 6 + m] * J[i * 6 + n];
			}
			A(m,n) = a;
		}
		double b = 0;
		for (int i = 0; i < 4 * active.size(); i++)
		{
			b += J[i * 6 + m] * (p_residual[i]);
		}
		B(m,0) = b;
	}
	//double beta = A.det;
	// perform elimination: solve Ax=B

    Eigen::Matrix<double,6,1> x;

    x = A.colPivHouseholderQr().solve(B);
	if ( B.isApprox(A*x,1e-2)) // Precision of isApprox
	{
        bool converged = true;
		for (int m = 0; m < 6; m++)
		{
			tr[m] += step_size * x(m,0);
			if (fabs( x(m,0) ) > eps)
				converged = false;
		}

        if (converged)
		    return CONVERGED;
        else
            return UPDATED;
	}
	else
	{
        std::cout << "colPivHouseholderQr() results FAILED, difference: " << (A*x - B).transpose() << std::endl;
		return FAILED;
	}

    return FAILED;
}


std::vector<int> VisualOdometryStereo::getRandomSample (int N,int num) {

    // std::random_device rd;
    // std::mt19937 gen(rd());

    // std::uniform_int_distribution<int> dis(0, N-1);
    std::vector<int> result;

    for (int i=0 ; i<num ; )
    {
        // int data = dis(gen);

        int data = rand() % N;

        bool duplicated = false;
        for ( auto e : result)
        {
            if (e == data)
            {
                duplicated = true;
                break;
            }
        }

        if (!duplicated)
        {
            result.push_back(data);
            i++;
        }
    }

    return result;
}

std::vector<int> VisualOdometryStereo::getInlier(vector<double> &tr)
{

	// mark all observations active, empty
	std::vector<int> active;

	// extract observations and compute predictions
	computeObservations(active, true);
	computeResidualsAndJacobian(tr, active, true);

	// compute inliers
    double threshold = param.inlier_threshold * param.inlier_threshold;
	std::vector<int> inliers;
	for (int i = 0; i < matches_quad_vec->size() ; i++)
    {
		double sq0 = p_residual[4 * i + 0] * p_residual[4 * i + 0];
		double sq1 = p_residual[4 * i + 1] * p_residual[4 * i + 1];
		double sq2 = p_residual[4 * i + 2] * p_residual[4 * i + 2];
		double sq3 = p_residual[4 * i + 3] * p_residual[4 * i + 3];
		if ( sq0 + sq1 + sq2 + sq3 < threshold)
            inliers.push_back(i);
        // double diff_lx = p_observe[4 * i + 0] - p_predict[4 * i + 0];
        // double diff_ly = p_observe[4 * i + 1] - p_predict[4 * i + 1];
        // double diff_rx = p_observe[4 * i + 2] - p_predict[4 * i + 2];
        // double diff_ry = p_observe[4 * i + 3] - p_predict[4 * i + 3];

        // if ( diff_lx*diff_lx + diff_ly*diff_ly + diff_rx*diff_rx + diff_ry*diff_ry < threshold)
        //     inliers.push_back(i);
    }

	return inliers;
}


void VisualOdometryStereo::computeResidualsAndJacobian(const std::vector<double> &tr, const std::vector<int> &active, bool inlier_mode)
{

    double &_cu = param.calib.cu;
    double &_cv = param.calib.cv;
    double &_f = param.calib.f;
    
	// extract motion parameters
	double rx = tr[0]; double ry = tr[1]; double rz = tr[2];
	double tx = tr[3]; double ty = tr[4]; double tz = tr[5];

	// precompute sine/cosine
	double sx = std::sin(rx); double cx = std::cos(rx); double sy = std::sin(ry);     // rx = alpha, ry = beta, rz = gamma
	double cy = std::cos(ry); double sz = std::sin(rz); double cz = std::cos(rz);

	// compute rotation matrix and derivatives
	// rotation matrix = Rz*Ry*Rx
	double r00 = +cy*cz;          double r01 = -cy*sz;          double r02 = +sy;
	double r10 = +sx*sy*cz + cx*sz; double r11 = -sx*sy*sz + cx*cz; double r12 = -sx*cy;
	double r20 = -cx*sy*cz + sx*sz; double r21 = +cx*sy*sz + sx*cz; double r22 = +cx*cy;

	double rdrx10 = +cx*sy*cz - sx*sz; double rdrx11 = -cx*sy*sz - sx*cz; double rdrx12 = -cx*cy;
	double rdrx20 = +sx*sy*cz + cx*sz; double rdrx21 = -sx*sy*sz + cx*cz; double rdrx22 = -sx*cy;
	double rdry00 = -sy*cz;          double rdry01 = +sy*sz;          double rdry02 = +cy;
	double rdry10 = +sx*cy*cz;       double rdry11 = -sx*cy*sz;       double rdry12 = +sx*sy;
	double rdry20 = -cx*cy*cz;       double rdry21 = +cx*cy*sz;       double rdry22 = -cx*sy;
	double rdrz00 = -cy*sz;          double rdrz01 = -cy*cz;
	double rdrz10 = -sx*sy*sz + cx*cz; double rdrz11 = -sx*sy*cz - cx*sz;
	double rdrz20 = +cx*sy*sz + sx*cz; double rdrz21 = +cx*sy*cz - sx*sz;

	// loop variables
	double X1cd, Y1cd, Z1cd;

	// for all observations do
	for (int i = 0; i < active.size() || (inlier_mode && i < matches_quad_vec->size() ); i++)
	{

		// get 3d point in previous coordinate system
		const double &X1p = X[ (inlier_mode ? i : active[i]) ];
		const double &Y1p = Y[ (inlier_mode ? i : active[i]) ];
		const double &Z1p = Z[ (inlier_mode ? i : active[i]) ];

		// compute 3d point in current left coordinate system
		double X1c = r00*X1p + r01*Y1p + r02*Z1p + tx;
		double Y1c = r10*X1p + r11*Y1p + r12*Z1p + ty;
		double Z1c = r20*X1p + r21*Y1p + r22*Z1p + tz;

		// weighting   hm: (centre points are given up to 20x of weight, only in x-axis)
		double weight = 1.0;
		if (param.reweighting)
			weight = 1.0 / (fabs(p_observe[4 * i + 0] - _cu) / fabs(_cu) + 0.05);   // only for current left image

		// compute 3d point in current right coordinate system
		double X2c = X1c - param.base;

		// for all paramters do  // six parameters: 3 rotations and 3 translations
		for (int j = 0; !inlier_mode && j < 6; j++)
		{

			// derivatives of 3d pt. in curr. left coordinates wrt. param j
			switch (j)
			{
			case 0: X1cd = 0;
				Y1cd = rdrx10*X1p + rdrx11*Y1p + rdrx12*Z1p;
				Z1cd = rdrx20*X1p + rdrx21*Y1p + rdrx22*Z1p;
				break;
			case 1: X1cd = rdry00*X1p + rdry01*Y1p + rdry02*Z1p;
				Y1cd = rdry10*X1p + rdry11*Y1p + rdry12*Z1p;
				Z1cd = rdry20*X1p + rdry21*Y1p + rdry22*Z1p;
				break;
			case 2: X1cd = rdrz00*X1p + rdrz01*Y1p;
				Y1cd = rdrz10*X1p + rdrz11*Y1p;
				Z1cd = rdrz20*X1p + rdrz21*Y1p;
				break;
			case 3: X1cd = 1; Y1cd = 0; Z1cd = 0; break;
			case 4: X1cd = 0; Y1cd = 1; Z1cd = 0; break;
			case 5: X1cd = 0; Y1cd = 0; Z1cd = 1; break;
			}

			// set jacobian entries (project via K)
			J[(4 * i + 0) * 6 + j] = weight*_f*(X1cd*Z1c - X1c*Z1cd) / (Z1c*Z1c); // left u'
			J[(4 * i + 1) * 6 + j] = weight*_f*(Y1cd*Z1c - Y1c*Z1cd) / (Z1c*Z1c); // left v'
			J[(4 * i + 2) * 6 + j] = weight*_f*(X1cd*Z1c - X2c*Z1cd) / (Z1c*Z1c); // right u'
			J[(4 * i + 3) * 6 + j] = weight*_f*(Y1cd*Z1c - Y1c*Z1cd) / (Z1c*Z1c); // right v'
		}

		// set prediction (project via K)
		p_predict[4 * i + 0] = _f*X1c / Z1c + _cu; // left center u
		p_predict[4 * i + 1] = _f*Y1c / Z1c + _cv; // left v
		p_predict[4 * i + 2] = _f*X2c / Z1c + _cu; // right u
		p_predict[4 * i + 3] = _f*Y1c / Z1c + _cv; // right v

		// set residuals
		p_residual[4 * i + 0] = weight*(p_observe[4 * i + 0] - p_predict[4 * i + 0]);
		p_residual[4 * i + 1] = weight*(p_observe[4 * i + 1] - p_predict[4 * i + 1]);
		p_residual[4 * i + 2] = weight*(p_observe[4 * i + 2] - p_predict[4 * i + 2]);
		p_residual[4 * i + 3] = weight*(p_observe[4 * i + 3] - p_predict[4 * i + 3]);

        // if (inlier_mode)
        // {
        //     std::cout << 4*i << ": p_residual[4*i]= " << p_residual[4*i] << std::endl;
        // }
	} 
}


Eigen::Matrix<double,4,4> VisualOdometryStereo::transformationVectorToMatrix( const std::vector<double> &tr ) {

  // extract parameters
  double rx = tr[0];
  double ry = tr[1];
  double rz = tr[2];
  double tx = tr[3];
  double ty = tr[4];
  double tz = tr[5];

  // precompute sine/cosine
  double sx = sin(rx);
  double cx = cos(rx);
  double sy = sin(ry);
  double cy = cos(ry);
  double sz = sin(rz);
  double cz = cos(rz);

  // compute transformation
  Eigen::Matrix<double,4,4> Tr;
  Tr(0,0) = +cy*cz;          Tr(0,1) = -cy*sz;          Tr(0,2) = +sy;    Tr(0,3) = tx;
  Tr(1,0) = +sx*sy*cz+cx*sz; Tr(1,1) = -sx*sy*sz+cx*cz; Tr(1,2) = -sx*cy; Tr(1,3) = ty;
  Tr(2,0) = -cx*sy*cz+sx*sz; Tr(2,1) = +cx*sy*sz+sx*cz; Tr(2,2) = +cx*cy; Tr(2,3) = tz;
  Tr(3,0) = 0;               Tr(3,1) = 0;               Tr(3,2) = 0;      Tr(3,3) = 1;
  return Tr;
}

#endif // VISO_STEREO_H