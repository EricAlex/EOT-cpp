#pragma once
#include "globaldef.h"
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/algorithms/intersection.hpp>
#include <boost/geometry/algorithms/area.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <random>

namespace bg = boost::geometry;
namespace bgm = bg::model;

typedef bgm::d2::point_xy<double> point_type;
typedef bgm::polygon<point_type> polygon_type;

class utilities{
    public:
      static Eigen::MatrixXd sampleWishart(int df, const Eigen::MatrixXd& S);
      static Eigen::MatrixXd sampleInverseWishart(int df, const Eigen::MatrixXd& S);
      static size_t mean_number_of_measurements(const Eigen::Vector2d& eigenvalues, float grid_resolution);
      static double multivariate_gamma(double a, size_t p);
      static double inverseWishartPDF(const Eigen::MatrixXd& X, int df, const Eigen::MatrixXd& S);
      static double sampleGaussian(double mu, double sigma);
      static double sampleUniform(double a, double b);
      static double samplePoisson(double lambda);
      static double gaussianPDF(double x, double mu, double sigma);
      static Eigen::VectorXd sampleMvNormal(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covar);
      static double mvnormPDF(const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sigma);
      static double discreteUniformPDF(size_t grid_c);
      static double gammaPDF(double x, double alpha, double beta);
      static double gammaPDF_mv(double x, double mean, double variance);
      static double sampleGamma(double alpha, double beta);
      static double sampleGamma_mv(double mean, double variance);
      static double p2lDistance(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& p);
      static double Q_function(const double x);
      static double error_function(const double x);
      static double Q_function_integral(const double x);
      static double measurement_likelihood_(const po_kinematic& x, const Eigen::Vector2d& eigenvalues, const Eigen::Matrix2d& eigenvectors, 
                                            const double gate_ratio, const Eigen::Vector2d& M, const double sd_noise);
      static bool isInPolygon(const vector<Eigen::Vector2d>& convex_hull,
                              const Eigen::Vector2d& test_point);
      static void extent2Polygon(const po_kinematic& x, 
                                 const Eigen::Vector2d& eigenvalues,
                                 const Eigen::Matrix2d& eigenvectors,
                                 const double ratio,
                                 vector<Eigen::Vector2d>& polygon);
      static Eigen::MatrixXd sqrtm(const Eigen::MatrixXd& A);
      static Eigen::MatrixXd eigen2Extent(const Eigen::VectorXd& eigenvalues, const Eigen::MatrixXd& eigenvectors);
      static double calculateIoU(const vector<Eigen::Vector2d>& poly1Points, const vector<Eigen::Vector2d>& poly2Points);
      static double calculateIoT(const vector<Eigen::Vector2d>& poly1Points, const vector<Eigen::Vector2d>& poly2Points);
      static double calculateIoS(const vector<Eigen::Vector2d>& poly1Points, const vector<Eigen::Vector2d>& poly2Points);
};