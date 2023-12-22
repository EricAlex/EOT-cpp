#include <iostream>
#include <unistd.h>
#include <unordered_set>
#include <boost/filesystem.hpp>

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

void generateTargetOutline(const Eigen::Vector4d& tracks, const Eigen::Matrix2d& extent, const size_t numMeasurements, 
                           const double measurementVariance, vector<Eigen::Vector2d>& measurements){
    po_kinematic tmpKine = {tracks(0), tracks(1), tracks(2), tracks(3), 0, 0};
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(extent);
    vector<Eigen::Vector2d> tmpPolygon;
    utilities::extent2Polygon(tmpKine, eigensolver.eigenvalues(), eigensolver.eigenvectors(), 1.0, tmpPolygon);
    tmpPolygon.push_back(tmpPolygon[0]);
    vector< pair<double, double> > range;
    double rangeSum(0);
    for(size_t i=0; i<(tmpPolygon.size()-1); ++i){
        double beforeAdd(rangeSum);
        Eigen::Vector2d edgeVec = tmpPolygon[i+1] - tmpPolygon[i];
        rangeSum += edgeVec.norm();
        range.push_back(make_pair(beforeAdd, rangeSum));
    }
    for(size_t m=0; m<(numMeasurements-1); ++m){
        double rdL = utilities::sampleUniform(0, rangeSum);
        size_t idx(0);
        for(size_t i=0; i<range.size(); ++i){
            if((rdL>=range[i].first)&&(rdL<=range[i].second)){
                idx = i;
                break;
            }
        }
        Eigen::Vector2d edgeVec = tmpPolygon[idx+1] - tmpPolygon[idx];
        Eigen::Vector2d measurementsTmp1 = tmpPolygon[idx] + ((rdL - range[idx].first)/edgeVec.norm()) * edgeVec 
            + sqrt(measurementVariance)*Eigen::Vector2d(utilities::sampleGaussian(0, 1), utilities::sampleGaussian(0, 1));
        measurements.push_back(measurementsTmp1);
    }
    measurements.push_back(Eigen::Vector2d(tracks(0), tracks(1))
        + sqrt(measurementVariance)*Eigen::Vector2d(utilities::sampleGaussian(0, 1), utilities::sampleGaussian(0, 1)));
}

void generateClutteredMeasurements(const vector< vector<Eigen::Vector4d> >& targetTracks, 
                                   const vector< vector<Eigen::Matrix2d> >& targetExtents,
                                   const eot_param& parameters, 
                                   const grid_para& grid_parameters, 
                                   targetShape shape,
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
                size_t numMeasurementsTmp = utilities::samplePoisson(parameters.meanMeasurements);
                if(shape == targetShape::ELLIPSE){
                    Eigen::Matrix2d covar = targetExtents[s][t]*targetExtents[s][t];
                    for(size_t m=0; m<numMeasurementsTmp; ++m){
                        Eigen::Vector2d measurementsTmp1 = Eigen::Vector2d(targetTracks[s][t](0), targetTracks[s][t](1)) + utilities::sampleMvNormal(Eigen::Vector2d(0.0, 0.0), covar) 
                        + sqrt(parameters.measurementVariance)*Eigen::Vector2d(utilities::sampleGaussian(0, 1), utilities::sampleGaussian(0, 1));
                        measurements.push_back(measurementsTmp1);
                    }
                }else if(shape == targetShape::RECTANGLE){
                    generateTargetOutline(targetTracks[s][t], targetExtents[s][t], numMeasurementsTmp, parameters.measurementVariance, measurements);
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
#if SIMULATION
int main(void){
    // parameters for simulations
    grid_para grid_parameters = {
        .dim1_min = -200,
        .dim1_max = 200,
        .dim2_min = -200,
        .dim2_max = 200,
        .grid_res = 0.5
    };
    double meanTargetDimension = 3;
    eot_param para = {
        .accelerationDeviation = 1,
        .rotationalAccelerationDeviation = 0.01,
        .survivalProbability = 0.999,
        .meanBirths = 0.001,
        .measurementVariance = grid_parameters.grid_res*grid_parameters.grid_res,
        .meanMeasurements = 25,
        .meanClutter = 5,
        .priorVelocityCovariance = Eigen::DiagonalMatrix<double, 2>(100, 100),
        .priorTurningRateDeviation = 0.01,
        .meanTargetDimension = meanTargetDimension,
        .meanPriorExtent = meanTargetDimension * Eigen::Matrix2d::Identity(),
        .priorExtentDegreeFreedom = 100,
        .degreeFreedomPrediction = 20000,
        .numParticles = 300,
        .regularizationDeviation = 0,
        .detectionThreshold = 0.5,
        .thresholdPruning = 1e-3,
        .numOuterIterations = 2
    };
    size_t numSteps = 60;
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
    generateClutteredMeasurements(targetTracks, targetExtents, para, grid_parameters, targetShape::RECTANGLE, clutteredMeasurements);
    EOT sim_eot;
    sim_eot.init(para);
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
            size.push_back(3);
        }
        // Plot line from given x and y data.
        auto l = matplot::scatter(x, y, size);
        l->marker_color("b");
        l->marker_face(true);
        matplot::hold(on);

        for(size_t t=0; t<targetTracks[s].size(); ++t){
            if(isnan(targetTracks[s][t](0))){
                continue;
            }
            po_kinematic tmpKine = {targetTracks[s][t](0), targetTracks[s][t](1), targetTracks[s][t](2), targetTracks[s][t](3), 0, 0};
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(targetExtents[s][t]);
            x.clear(); y.clear();
            vector<Eigen::Vector2d> tmpPolygon;
            utilities::extent2Polygon(tmpKine, eigensolver.eigenvalues(), eigensolver.eigenvectors(), 1.0, tmpPolygon);
            for(size_t v=0; v<tmpPolygon.size(); ++v){
                x.push_back(tmpPolygon[v](0));
                y.push_back(tmpPolygon[v](1));
            }
            x.push_back(tmpPolygon[0](0));
            y.push_back(tmpPolygon[0](1));
            matplot::plot(x, y, "g")->line_width(3);
        }

        if(potential_objects_out.size()>0){
            for(size_t t=0; t<potential_objects_out.size(); ++t){
                x.clear(); y.clear();
                vector<Eigen::Vector2d> tmpPolygon;
                utilities::extent2Polygon(potential_objects_out[t].kinematic, potential_objects_out[t].extent.eigenvalues, 
                                          potential_objects_out[t].extent.eigenvectors, 1.0, tmpPolygon);
                for(size_t v=0; v<tmpPolygon.size(); ++v){
                    x.push_back(tmpPolygon[v](0));
                    y.push_back(tmpPolygon[v](1));
                }
                x.push_back(tmpPolygon[0](0));
                y.push_back(tmpPolygon[0](1));
                matplot::plot(x, y, "r")->line_width(3);
            }
        }

        matplot::hold(off);

        // Set x-axis and y-axis
        matplot::xlim({-startRadius-10, startRadius+10});
        matplot::ylim({-startRadius-10, startRadius+10});
        usleep(1e5);
    }
    return 0;
}
#else
bool FNcmp(boost::filesystem::path path1, boost::filesystem::path path2){
    string s1 = path1.stem().string();
    string s2 = path2.stem().string();
    double n1 = atoi(s1.c_str());
    double n2 = atoi(s2.c_str());
    return n1 < n2;
}
vector<boost::filesystem::path> streamFile(string dataPath){
    vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath},
                                          boost::filesystem::directory_iterator{});
    string extension = ".bin";
    vector<boost::filesystem::path> paths_out;
	for (auto iter=paths.begin(); iter != paths.end(); ++iter){
        if(iter->extension().string() == extension){
            paths_out.push_back(iter->string());
        }
	}
    sort(paths_out.begin(), paths_out.end(), FNcmp);
    return paths_out;
}
void loadData(const boost::filesystem::path& pcd_path, 
              vector<measurement>& Measurements,
              grid_para& grid_parameters,
              double& scanTime,
              uint64_t& frame_idx){
    ifstream fbinaryin(pcd_path.string().c_str(), ios::in|ios::binary);
    if(fbinaryin.is_open()){
        size_t num_M;
        fbinaryin.read(reinterpret_cast<char*>(&num_M), sizeof(num_M));
        for(size_t i=0; i<num_M; ++i){
            measurement tmpM;
            fbinaryin.read(reinterpret_cast<char*>(&(tmpM(0))), sizeof(tmpM(0)));
            fbinaryin.read(reinterpret_cast<char*>(&(tmpM(1))), sizeof(tmpM(1)));
            Measurements.push_back(tmpM);
        }
        fbinaryin.read(reinterpret_cast<char*>(&(grid_parameters.dim1_min)), sizeof(grid_parameters.dim1_min));
        fbinaryin.read(reinterpret_cast<char*>(&(grid_parameters.dim1_max)), sizeof(grid_parameters.dim1_max));
        fbinaryin.read(reinterpret_cast<char*>(&(grid_parameters.dim2_min)), sizeof(grid_parameters.dim2_min));
        fbinaryin.read(reinterpret_cast<char*>(&(grid_parameters.dim2_max)), sizeof(grid_parameters.dim2_max));
        fbinaryin.read(reinterpret_cast<char*>(&(grid_parameters.grid_res)), sizeof(grid_parameters.grid_res));
        fbinaryin.read(reinterpret_cast<char*>(&scanTime), sizeof(scanTime));
        fbinaryin.read(reinterpret_cast<char*>(&frame_idx), sizeof(frame_idx));
        fbinaryin.close();
    }else{
        cout<<"ERROR: can not open file "<<pcd_path.string()<<endl;
    }
}
int main(int argc, char *argv[]){
    if((argc>2)&&(strlen(argv[1])==2)&&(argv[1][0]=='-')&&(argv[1][1]=='i')){
        string path(argv[2]);
        vector<boost::filesystem::path> stream = streamFile(path);
        for(auto it=stream.begin(); it!=stream.end(); ++it){
            vector<measurement> Measurements;
            grid_para grid_parameters;
            double scanTime;
            uint64_t frame_idx;
            loadData(*it, Measurements, grid_parameters, scanTime, frame_idx);
            cout<<"Measurements.szie(): "<<Measurements.size()<<endl;
            usleep(1e5);
        }
    }
    return 0;
}
#endif