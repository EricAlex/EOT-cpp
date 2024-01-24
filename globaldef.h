#pragma once
#include <Eigen/Dense>

using namespace std;

#define DEBUG	true
#define SIMULATION	true

#define LIDAR_ANGULAR_RESOLUTION	0.0013962634 // in radian
#define EOT_INIT_VAILD_GRID_LABEL	1

enum targetShape{
	ELLIPSE,
	RECTANGLE
};

struct po_kinematic{
	double p1; // absolute position_1
	double p2; // absolute position_2
	// double rtv_p1; // relative position_1
	// double rtv_p2; // relative position_2
	double v1; // absolute spped_1
	double v2; // absolute spped_2
	double t; // absolute turning rate in radian per second
	double s; // mean number of generated measurements
};

struct po_extent{
	Eigen::Matrix2d e;
	Eigen::Vector2d eigenvalues;
	Eigen::Matrix2d eigenvectors;
};

struct po_label{
	uint64_t frame_idx;
	uint64_t label_v;
};

struct PO{
	po_kinematic kinematic;
	po_extent extent;
	po_label label;
};

typedef Eigen::Vector2d measurement;

struct vec2d {
    double data[2];
    double operator[](int idx) const { return data[idx]; }
};

struct grid_para{
	double dim1_min;
	double dim1_max;
	double dim2_min;
	double dim2_max;
	double grid_res;
};

struct eot_param{
	// main parameters of the statistical model
	double accelerationDeviation; // acceleration deviation
	double rotationalAccelerationDeviation; // angular acceleration deviation
	double survivalProbability; // default 0.999
	double meanBirths; // default 0.01
	double measurementVariance; // default (grid size)^2
	size_t meanMeasurements;
	size_t meanClutter;
	// prior distribution parameters
	Eigen::Matrix2d priorVelocityCovariance; // default diag(5^2, 5^2)
	double priorTurningRateDeviation; // default 0.01 rad/s
	double meanTargetDimension;
	Eigen::Matrix2d meanPriorExtent; // default 3*I
	double priorExtentDegreeFreedom; // default 30
	double degreeFreedomPrediction; // default 2000
	// sampling parameters
	size_t numParticles; // default 1000
	double regularizationDeviation; // default 0
	// detection and pruning parameters
	double detectionThreshold; // default 0.5
	double thresholdPruning; // default 10^(-3)
	// message passing parameters
	size_t numOuterIterations; // default 3
};