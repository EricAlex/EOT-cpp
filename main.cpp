#include <iostream>
#include <unistd.h>

#include <matplot/matplot.h>

#include "globaldef.h"
#include "utilities.h"
#include "EOT.h"

using namespace matplot;
using namespace std;

void getStartStates(const size_t numTargets, 
                    const double startRadius, 
                    const double startVelocity,
                    const eot_param& parameters,
                    vector<Eigen::Vector4d>& startStates,
                    vector<Eigen::Matrix2d>& startMatrixes){
    Eigen::MatrixXd priorExtentShape = parameters.meanPriorExtent*(parameters.priorExtentDegreeFreedom-parameters.meanPriorExtent.cols()-1);
    startMatrixes.resize(numTargets);
    for(size_t t=0; t<numTargets; ++t){
        startMatrixes[t] = utilities::sampleInverseWishart(parameters.priorExtentDegreeFreedom, priorExtentShape);
    }
    startStates.resize(numTargets);
    if(numTargets == 1){
        startStates[0] = Eigen::Vector4d(0, 0, startVelocity, 0);
    }else if(numTargets > 1){
        startStates[0] = Eigen::Vector4d(0, startRadius, 0, -startVelocity);
        double stepSize = 2*M_PI/numTargets;
        double angle(0.0);
        for(size_t t=1; t<numTargets; ++t){
            angle += stepSize;
            startStates[t] = Eigen::Vector4d(sin(angle)*startRadius, cos(angle)*startRadius, -sin(angle)*startVelocity, -cos(angle)*startVelocity);
        }
    }
}

void generateTracksUnknown(const eot_param& parameters, 
                           const vector<Eigen::Vector4d>& startStates, 
                           const vector<Eigen::Matrix2d>& startMatrixes, 
                           const vector<pair<size_t, size_t>>& appearanceFromTo, 
                           const size_t numSteps,
                           const double scanTime,
                           vector< vector<Eigen::Vector4d> >& targetTracks,
                           vector< vector<Eigen::Matrix2d> >& targetExtents){
    double nanValue = nan("");
    size_t numTargets = startStates.size();
    targetTracks.resize(numSteps, vector<Eigen::Vector4d>(numTargets, (Eigen::Vector4d() << nanValue, nanValue, nanValue, nanValue).finished()));
    targetExtents.resize(numSteps, vector<Eigen::Matrix2d>(numTargets, (Eigen::Matrix2d() << nanValue, nanValue, nanValue, nanValue).finished()));
    for(size_t t=0; t<numTargets; ++t){
        Eigen::Vector4d tmp = startStates[t];
        for(size_t step=0; step<numSteps; ++step){
            double r1 = utilities::sampleGaussian(0, parameters.accelerationDeviation);
            double r2 = utilities::sampleGaussian(0, parameters.accelerationDeviation);
            tmp(0) += (tmp(2)*scanTime + 0.5*scanTime*scanTime*r1);
            tmp(1) += (tmp(3)*scanTime + 0.5*scanTime*scanTime*r2);
            tmp(2) += r1*scanTime;
            tmp(3) += r2*scanTime;
            if((step >= appearanceFromTo[t].first)&&(step <= appearanceFromTo[t].second)){
                targetTracks[step][t] = tmp;
                targetExtents[step][t] = startMatrixes[t];
            }
        }
    }
}

void generateClutteredMeasurements(const vector< vector<Eigen::Vector4d> >& targetTracks, 
                                   const vector< vector<Eigen::Matrix2d> >& targetExtents,
                                   const eot_param& parameters, 
                                   const grid_para& grid_parameters, 
                                   vector< vector<Eigen::Vector2d> >& clutteredMeasurements){
    if((targetTracks.size()!=targetExtents.size())||(targetTracks.size()<=0)){
        return;
    }else{
        size_t numSteps = targetTracks.size();
        size_t numTargets = targetTracks[0].size();
        clutteredMeasurements.resize(numSteps);
        for(size_t s=0; s<numSteps; ++s){
            vector<Eigen::Vector2d> measurements;
            for(size_t t=0; t<numTargets; ++t){
                if(isnan(targetTracks[s][t](0))){
                    continue;
                }
                Eigen::Matrix2d covar = targetExtents[s][t]*targetExtents[s][t];
                size_t numMeasurementsTmp = utilities::samplePoisson(parameters.meanMeasurements);
                for(size_t m=0; m<numMeasurementsTmp; ++m){
                    Eigen::Vector2d measurementsTmp1 = Eigen::Vector2d(targetTracks[s][t](0), targetTracks[s][t](1)) + utilities::sampleMvNormal(Eigen::Vector2d(0.0, 0.0), covar) 
                    + sqrt(parameters.measurementVariance)*Eigen::Vector2d(utilities::sampleGaussian(0, 1), utilities::sampleGaussian(0, 1));
                    measurements.push_back(measurementsTmp1);
                }
            }
            size_t numFalseAlarms = utilities::samplePoisson(parameters.meanClutter);
            for(size_t m=0; m<numFalseAlarms; ++m){
                measurements.push_back(Eigen::Vector2d(grid_parameters.dim1_min+utilities::sampleUniform(0, 1)*(grid_parameters.dim1_max-grid_parameters.dim1_min),
                                                       grid_parameters.dim2_min+utilities::sampleUniform(0, 1)*(grid_parameters.dim2_max-grid_parameters.dim2_min)));
            }
            std::mt19937 g(std::random_device{}());
            std::shuffle(measurements.begin(), measurements.end(), g);
            clutteredMeasurements[s] = measurements;
        }
    }
}

int main(void){
    // parameters for simulations
    eot_param para = {
        .accelerationDeviation = 1,
        .rotationalAccelerationDeviation = 0.01,
        .survivalProbability = 0.99,
        .meanBirths = 0.01,
        .surveillanceAreaSize = 160000,
        .measurementVariance = 1,
        .meanMeasurements = 8,
        .meanClutter = 10,
        .priorVelocityCovariance = Eigen::DiagonalMatrix<double, 2>(100, 100),
        .priorTurningRateDeviation = 0.01,
        .meanPriorExtent = 3 * Eigen::Matrix2d::Identity(),
        .priorExtentDegreeFreedom = 100,
        .degreeFreedomPrediction = 20000,
        .numParticles = 1000,
        .regularizationDeviation = 0,
        .detectionThreshold = 0.5,
        .thresholdPruning = 1e-3,
        .numOuterIterations = 2
    };
    grid_para grid_parameters = {
        .dim1_min = -200,
        .dim1_max = 200,
        .dim2_min = -200,
        .dim2_max = 200,
        .grid_res = 1
    };
    size_t numSteps = 50;
    size_t numTargets = 5;
    double startRadius = 75;
    double startVelocity = 10;
    double scanTime = 0.2;
    vector<Eigen::Vector4d> startStates;
    vector<Eigen::Matrix2d> startMatrixes;
    getStartStates(numTargets, startRadius, startVelocity, para, startStates, startMatrixes);
    vector<pair<size_t, size_t>> appearanceFromTo = {make_pair(3,83), make_pair(3,83), make_pair(6,86), make_pair(6,86), make_pair(9,89), make_pair(9,89), 
        make_pair(12,92), make_pair(12,92), make_pair(15,95), make_pair(15,95)};
    vector< vector<Eigen::Vector4d> > targetTracks;
    vector< vector<Eigen::Matrix2d> > targetExtents;
    generateTracksUnknown(para, startStates, startMatrixes, appearanceFromTo, numSteps, scanTime, targetTracks, targetExtents);
    vector< vector<Eigen::Vector2d> > clutteredMeasurements;
    generateClutteredMeasurements(targetTracks, targetExtents, para, grid_parameters, clutteredMeasurements);
    EOT sim_eot(para);
    for(size_t s=0; s<numSteps; ++s){
        cout<<"Number of measurements: "<<clutteredMeasurements[s].size()<<endl;
        vector<PO> potential_objects_out;
        sim_eot.eot_track(clutteredMeasurements[s], grid_parameters, scanTime, s, potential_objects_out);
        cout<<"potential_objects_out.size(): "<<potential_objects_out.size()<<endl;
        // plot result
        vector<double> x, y, size;
        for(size_t m=0; m<clutteredMeasurements[s].size(); ++m){
            x.push_back(clutteredMeasurements[s][m](0));
            y.push_back(clutteredMeasurements[s][m](1));
            size.push_back(4);
        }
        // Plot line from given x and y data.
        auto l = matplot::scatter(x, y, size);
        l->marker_face_color("b");
        l->marker_face(true);
        matplot::hold(on);

        if(potential_objects_out.size()>0){
            x.clear(); y.clear(); size.clear();
            for(size_t t=0; t<potential_objects_out.size(); ++t){
                x.push_back(potential_objects_out[t].kinematic.p1);
                y.push_back(potential_objects_out[t].kinematic.p2);
                size.push_back(12);
            }
            auto l_o = matplot::scatter(x, y, size);
            l_o->marker_face_color("r");
            l_o->marker_face(true);
        }
        matplot::hold(off);

        // Set x-axis and y-axis
        matplot::xlim({-startRadius, startRadius});
        matplot::ylim({-startRadius, startRadius});
        usleep(1e5);
    }
    return 0;
}