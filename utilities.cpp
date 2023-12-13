#include "utilities.h"

Eigen::MatrixXd utilities::sampleWishart(int df, const Eigen::MatrixXd& S) {
    int dim = S.cols();

    static std::random_device rd;
    static std::mt19937 rng(rd());
    std::normal_distribution<double> nd(0, 1);
    // Build N_{ij}
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(dim, dim);
    for (int j = 0; j < dim; j++) {
        for (int i = 0; i < j; i++) {
            N(i, j) = nd(rng);
        }
    }
    // Build V_j
    std::vector<double> V(dim);
    for (int i = 0; i < dim; i++) {
        std::gamma_distribution<double> gd((df-i+0.0)/2, 2);
        V[i] = gd(rng);
    }
    // Build B
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(dim, dim);
    // b_{11} = V_1 (first j, where sum = 0 because i == j and the inner
    //               loop is never entered).
    // b_{jj} = V_j + \sum_{i=1}^{j-1} N_{ij}^2, j = 2, 3, ..., p
    for (int j = 0; j < dim; j++) {
        double sum = 0;
        for (int i = 0; i < j; i++) {
            sum += N(i, j)*N(i, j);
        }
        B(j, j) = V[j] + sum;
    }
    // b_{1j} = N_{1j} * \sqrt V_1
    for (int j = 1; j < dim; j++) {
        B(0, j) = N(0, j)*sqrt(V[0]);
        B(j, 0) = B(0, j);
    }
    // b_{ij} = N_{ij} * \sqrt V_1 + \sum_{k=1}^{i-1} N_{ki}*N_{kj}
    for (int j = 1; j < dim; j++) {
        for (int i = 1; i < j; i++) {
            double sum = 0;
            for (int k = 0; k < i; k++) {
                sum += N(k, i) * N(k, j);
            }
            B(i, j) = N(i, j) * sqrt(V[i]) + sum;
            B(j, i) = B(i, j);
        }
    }
    Eigen::MatrixXd L = S.inverse().llt().matrixL();
    return L*B*L.transpose();
}

Eigen::MatrixXd utilities::sampleInverseWishart(int df, const Eigen::MatrixXd& S) {
    Eigen::MatrixXd A = sampleWishart(df, S);
    return A.inverse();
}

double utilities::multivariate_gamma(double a, size_t p){
    double result = std::pow(M_PI, p*(double(p)-1)/4);
    for(size_t j = 1; j <= p; j++){
        result *= boost::math::tgamma(a + (1-double(j))/2);
    }
    return result;
}

double utilities::inverseWishartPDF(const Eigen::MatrixXd& X, int df, const Eigen::MatrixXd& S){
    size_t dim = S.cols();
    double p(dim);
    double v(df);
    return std::pow(S.determinant(), v/2) * std::pow(X.determinant(), -(v+p+1)/2) 
    * std::exp(-0.5*(S*X.inverse()).trace()) / (std::pow(2, v*p/2)*multivariate_gamma(v/2, dim));
}

double utilities::sampleGaussian(double mu, double sigma){
    static std::random_device nd_rd;
    static std::mt19937 nd_rng(nd_rd());
    std::normal_distribution<> nd(mu, sigma);
    return nd(nd_rng);
}

// Uniform real distribution between a and b
double utilities::sampleUniform(double a, double b){
    static std::random_device ud_rd;
    static std::mt19937 ud_rng(ud_rd());
    std::uniform_real_distribution<> urd(a, b);
    return urd(ud_rng);
}

double utilities::samplePoisson(double lambda){
    static std::random_device pd_rd;
    static std::mt19937 pd_rng(pd_rd());
    std::poisson_distribution<int> pd(lambda);
    return pd(pd_rng);
}

double utilities::gaussianPDF(double x, double mu, double sigma){
    static const double sqrt_2pi = std::sqrt(2.0 * M_PI);
    return std::exp(-0.5*std::pow((x-mu)/sigma,2))/(sigma*sqrt_2pi);
}

Eigen::VectorXd utilities::sampleMvNormal(const Eigen::VectorXd& mean, 
                                          const Eigen::MatrixXd& covar) {
    // Perform Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> lltOfA(covar);
    Eigen::MatrixXd L = lltOfA.matrixL();

    // Generate standard normal random values
    Eigen::VectorXd stdNorm = Eigen::VectorXd(mean.size()).unaryExpr([](double dummy){return sampleGaussian(0.0, 1.0);});

    // Scale with Cholesky factor and shift by mean
    return mean + L*stdNorm;
}

double utilities::mvnormPDF(const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sigma) {
    int k = x.size();
    Eigen::VectorXd dist = x - mu;
    return std::exp(-0.5 * dist.transpose() * sigma.inverse() * dist)/std::sqrt(std::pow(2*M_PI, k) * sigma.determinant());
}

double utilities::discreteUniformPDF(size_t grid_c){
    return 1.0/double(grid_c);
}

// alpha: shape parameter, beta: inverse scale parameter
// mean: alpha/beta, variance: alpha/(beta^2)
double utilities::gammaPDF(double x, double alpha, double beta){
    return std::pow(beta, alpha)*std::pow(x, alpha-1)*std::exp(-beta*x)/std::tgamma(alpha);
}

double utilities::gammaPDF_mv(double x, double mean, double variance){
    double beta = mean/variance;
    double alpha = mean*beta;
    return gammaPDF(x, alpha, beta);
}

// alpha: shape parameter, beta: inverse scale parameter
double utilities::sampleGamma(double alpha, double beta){
    static std::random_device gd_rd;
    static std::mt19937 gd_rng(gd_rd());
    std::gamma_distribution<double> gd(alpha, 1/beta);
    return gd(gd_rng);
}

double utilities::sampleGamma_mv(double mean, double variance){
    double beta = mean/variance;
    double alpha = mean*beta;
    return sampleGamma(alpha, beta);
}

size_t utilities::mean_number_of_measurements(const Eigen::Vector2d& eigenvalues,
                                              float grid_resolution){
    return size_t((eigenvalues(0)+eigenvalues(1))/(2*grid_resolution)) + size_t(eigenvalues(0)*eigenvalues(1)/(grid_resolution*grid_resolution))/20 + 2;
}

double utilities::p2lDistance(const Eigen::Vector2d& a, 
                              const Eigen::Vector2d& b, 
                              const Eigen::Vector2d& p){
    Eigen::Vector2d v1 = b - a;
    Eigen::Vector2d v2 = p - b;
    return v2.norm()*sin(acos(fabs(v1.dot(v2))/(v1.norm()*v2.norm())));
}

double utilities::Q_function(const double x){
  return exp(-(x*x)/2)/12 + exp(-2*(x*x)/3)/4;
}

double utilities::measurement_likelihood_(const po_kinematic& x, 
                                          const Eigen::Vector2d& eigenvalues,
                                          const Eigen::Matrix2d& eigenvectors,
                                          const double gate_ratio,
                                          const Eigen::Vector2d& M, 
                                          const double sd_noise){
    double radius(gate_ratio * sqrt(std::pow(eigenvalues(1), 2) + std::pow(eigenvalues(0), 2)));
    if((M(0)<(x.p1-radius))||(M(0)>(x.p1+radius))||(M(1)<(x.p2-radius))||(M(1)>(x.p2+radius))){
        return 0;
    }else{
        double width(eigenvalues(0)), length(eigenvalues(1));
        Eigen::Vector2d V_w = eigenvectors.col(0);
        Eigen::Vector2d V_l = eigenvectors.col(1);
        Eigen::Vector2d half_w_vec = 0.5*width*V_w.normalized();
        Eigen::Vector2d half_l_vec = 0.5*length*V_l.normalized();
        Eigen::Vector2d P(x.p1, x.p2);
        double d1 = p2lDistance(P + half_w_vec + half_l_vec, P + half_w_vec - half_l_vec, M);
        double d2 = p2lDistance(P - half_w_vec + half_l_vec, P - half_w_vec - half_l_vec, M);
        double Q1 = Q_function(d1/sd_noise);
        double Q2 = Q_function(d2/sd_noise);
        double f1(0), f2(0);
        if((width>d1)&&(width>d2)){
            f1 = 1.0 - Q1 - Q2;
        }else if((d2>=width)&&(d2>=d1)){
            f1 = Q1 - Q2;
        }else if((d1>=width)&&(d1>=d2)){
            f1 = Q2 - Q1;
        }
        d1 = p2lDistance(P + half_l_vec + half_w_vec, P + half_l_vec - half_w_vec, M);
        d2 = p2lDistance(P - half_l_vec + half_w_vec, P - half_l_vec - half_w_vec, M);
        Q1 = Q_function(d1/sd_noise);
        Q2 = Q_function(d2/sd_noise);;
        if((length>d1)&&(length>d2)){
            f2 = 1.0 - Q1 - Q2;
        }else if((d2>=length)&&(d2>=d1)){
            f2 = Q1 - Q2;
        }else if((d1>=length)&&(d1>=d2)){
            f2 = Q2 - Q1;
        }
        return f1*f2/(length*width);
    }
}

bool utilities::isInPolygon(const vector<Eigen::Vector2d>& convex_hull,
                            const Eigen::Vector2d& test_point) {
  int i, j;
  bool c = false;
  for (i = 0, j = convex_hull.size() - 1; (size_t)i < convex_hull.size(); j = i++) {
    if (((convex_hull[i](1) > test_point(1)) != (convex_hull[j](1) > test_point(1))) &&
        (test_point(0) < (convex_hull[j](0) - convex_hull[i](0)) * (test_point(1) - convex_hull[i](1)) / (convex_hull[j](1) - convex_hull[i](1)) + convex_hull[i](0)))
      c = !c;
  }
  return c;
}

void utilities::extent2Polygon(const po_kinematic& x, 
                               const Eigen::Vector2d& eigenvalues,
                               const Eigen::Matrix2d& eigenvectors,
                               const double ratio,
                               vector<Eigen::Vector2d>& polygon){
    double width(ratio*eigenvalues(0)), length(ratio*eigenvalues(1));
    Eigen::Vector2d V_w = eigenvectors.col(0);
    Eigen::Vector2d V_l = eigenvectors.col(1);
    Eigen::Vector2d half_w_vec = 0.5*width*V_w.normalized();
    Eigen::Vector2d half_l_vec = 0.5*length*V_l.normalized();
    Eigen::Vector2d P(x.p1, x.p2);
    polygon.push_back(P + half_w_vec + half_l_vec);
    polygon.push_back(P + half_w_vec - half_l_vec);
    polygon.push_back(P - half_w_vec - half_l_vec);
    polygon.push_back(P - half_w_vec + half_l_vec);
}