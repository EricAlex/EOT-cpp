#include <iostream>
#include <unistd.h>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <matplot/matplot.h>

#include "globaldef.h"
#include "utilities.h"
#include "EOT.h"
#include "third_party/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

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
    // measurements.push_back(Eigen::Vector2d(tracks(0), tracks(1))
    //     + sqrt(measurementVariance)*Eigen::Vector2d(utilities::sampleGaussian(0, 1), utilities::sampleGaussian(0, 1)));
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

void generateClutteredMeasurements_static_test(const vector< vector<Eigen::Vector4d> >& targetTracks, 
                                               vector< vector<Eigen::Vector2d> >& clutteredMeasurements){
    size_t numSteps = targetTracks.size();
    clutteredMeasurements.resize(numSteps);
    float mLengthStep(0.5);
    vector<Eigen::Vector2d> tmpPolygon;
    // tmpPolygon.push_back(Eigen::Vector2d(-1, 2));
    // tmpPolygon.push_back(Eigen::Vector2d(1, 2));
    tmpPolygon.push_back(Eigen::Vector2d(1, -2));
    tmpPolygon.push_back(Eigen::Vector2d(-1, -2));
    tmpPolygon.push_back(Eigen::Vector2d(-1, 2));
    double rot_ang(3*M_PI/180);
    // rot_ang = 0;
    for(size_t s=0; s<numSteps; ++s){
        rot_ang += M_PI/180/5;
        Eigen::Matrix2d rot_mat;
        rot_mat << cos(rot_ang), -sin(rot_ang),
                    sin(rot_ang), cos(rot_ang);
        for(size_t i=0; i<tmpPolygon.size(); ++i){
            tmpPolygon[i] = rot_mat * tmpPolygon[i];
        }
        vector<Eigen::Vector2d> measurements;
        for(size_t i=0; i<(tmpPolygon.size()-1); ++i){
            Eigen::Vector2d edgeVec = tmpPolygon[i+1] - tmpPolygon[i];
            float edgeL = edgeVec.norm();
            edgeVec.normalize();
            for(float p=0; p<edgeL; p+=mLengthStep){
                measurements.push_back(tmpPolygon[i] + p*edgeVec);
            }
        }
        measurements.push_back(Eigen::Vector2d(0, 0));
        clutteredMeasurements[s] = measurements;
    }
}
#if SIMULATION
int main(void){
    #define STATIC_SIMULATION false
    #define EXPORT_FIGURE	false
    // parameters for simulations
    grid_para grid_parameters = {
        .dim1_min = -200,
        .dim1_max = 200,
        .dim2_min = -200,
        .dim2_max = 200,
        .grid_res = 0.5
    };
    double meanTargetDimension = 3;
    double measurementDeviation = meanTargetDimension/30;
    // double measurementDeviation = grid_parameters.grid_res;
    eot_param para = {
        .accelerationDeviation = 1,
        #if STATIC_SIMULATION
            .rotationalAccelerationDeviation = 30*M_PI/180,
        #else
            .rotationalAccelerationDeviation = M_PI/180,
        #endif
        .survivalProbability = 0.999,
        .meanBirths = 0.001,
        .measurementVariance = measurementDeviation*measurementDeviation,
        .meanMeasurements = 15,
        .meanClutter = 5,
        .priorVelocityCovariance = Eigen::DiagonalMatrix<double, 2>(100, 100),
        #if STATIC_SIMULATION
            .priorTurningRateDeviation = 30*M_PI/180,
        #else
            .priorTurningRateDeviation = M_PI/180,
        #endif
        .meanTargetDimension = meanTargetDimension,
        .meanPriorExtent = meanTargetDimension * Eigen::Matrix2d::Identity(),
        .priorExtentDegreeFreedom = 30,
        .degreeFreedomPrediction = 1000,
        .numParticles = 800,
        .ratioLegacyParticles = 0.4,
        .regularizationDeviation = meanTargetDimension/10,
        .detectionThreshold = 0.5,
        .thresholdPruning = 1e-3,
        .numOuterIterations = 2
    };
    #if STATIC_SIMULATION
        size_t numSteps = 100;
    #else
        size_t numSteps = 80;
    #endif
    size_t numTargets = 7;
    double startRadius = 90;
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
    #if STATIC_SIMULATION
        generateClutteredMeasurements_static_test(targetTracks, clutteredMeasurements);
    #else
        generateClutteredMeasurements(targetTracks, targetExtents, para, grid_parameters, targetShape::RECTANGLE, clutteredMeasurements);
    #endif
    EOT sim_eot;
    sim_eot.init(para);
    auto h = matplot::figure();
    h->size(1200, 1200);
    for(size_t s=0; s<numSteps; ++s){
        vector<PO> potential_objects_out;
        sim_eot.eot_track(clutteredMeasurements[s], grid_parameters, scanTime, s, potential_objects_out);
        // cout<<"potential_objects_out.size(): "<<potential_objects_out.size()<<endl;
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
        
        #if !STATIC_SIMULATION
            // plot GT
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
        #endif

        // plot result
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

                x.clear(); y.clear();
                x.push_back(potential_objects_out[t].kinematic.p1);
                y.push_back(potential_objects_out[t].kinematic.p2);
                x.push_back(potential_objects_out[t].kinematic.p1 + potential_objects_out[t].kinematic.v1);
                y.push_back(potential_objects_out[t].kinematic.p2 + potential_objects_out[t].kinematic.v2);
                matplot::plot(x, y, "r")->line_width(1);
            }
        }

        matplot::hold(off);

        // Set x-axis and y-axis
        #if STATIC_SIMULATION
            matplot::xlim({-5, 5});
            matplot::ylim({-5, 5});
        #else
            matplot::xlim({-startRadius-10, startRadius+10});
            matplot::ylim({-startRadius-10, startRadius+10});
        #endif

        matplot::title(std::to_string(s));

        #if EXPORT_FIGURE
            matplot::save("export/"+std::to_string(s)+".jpg");
        #endif
        usleep(1e5);
    }
    return 0;
}
#else
bool FNcmp(boost::filesystem::path path1, boost::filesystem::path path2){
    std::string s1 = path1.stem().string();
    std::string s2 = path2.stem().string();
    // Splitting by underscore and getting first component
    auto pos1 = s1.find('_');
    auto pos2 = s2.find('_');
    if (pos1 != std::string::npos) {
        s1 = s1.substr(0, pos1);
    }
    if (pos2 != std::string::npos) {
        s2 = s2.substr(0, pos2);
    }
    // Safely converting to numbers
    double n1, n2;
    std::istringstream(s1) >> n1;
    std::istringstream(s2) >> n2;
    // Check if conversion was successful
    if (std::istringstream(s1).fail() || std::istringstream(s2).fail()) {
        return false; // Or throw an exception, depending on your error handling strategy
    }
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
              vector<size_t>& promissing_new_t_idx,
              vector<Eigen::Matrix2d>& p_n_t_eigenvectors,
              vector<Eigen::Vector2d>& p_n_t_extents,
              Eigen::Matrix4d& pose4Predict,
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
        size_t num_new_tar;
        fbinaryin.read(reinterpret_cast<char*>(&num_new_tar), sizeof(num_new_tar));
        for(size_t i=0; i<num_new_tar; ++i){
            size_t idx;
            fbinaryin.read(reinterpret_cast<char*>(&(idx)), sizeof(idx));
            promissing_new_t_idx.push_back(idx);
        }
        for(size_t i=0; i<num_new_tar; ++i){
            Eigen::Matrix2d eigenvectors;
            fbinaryin.read(reinterpret_cast<char*>(&(eigenvectors(0, 0))), sizeof(eigenvectors(0, 0)));
            fbinaryin.read(reinterpret_cast<char*>(&(eigenvectors(1, 0))), sizeof(eigenvectors(1, 0)));
            fbinaryin.read(reinterpret_cast<char*>(&(eigenvectors(0, 1))), sizeof(eigenvectors(0, 1)));
            fbinaryin.read(reinterpret_cast<char*>(&(eigenvectors(1, 1))), sizeof(eigenvectors(1, 1)));
            p_n_t_eigenvectors.push_back(eigenvectors);
        }
        for(size_t i=0; i<num_new_tar; ++i){
            Eigen::Vector2d extents;
            fbinaryin.read(reinterpret_cast<char*>(&(extents(0))), sizeof(extents(0)));
            fbinaryin.read(reinterpret_cast<char*>(&(extents(1))), sizeof(extents(1)));
            p_n_t_extents.push_back(extents);
        }
        for(size_t row=0; row<4; ++row){
            for(size_t col=0; col<4; ++col){
                fbinaryin.read(reinterpret_cast<char*>(&(pose4Predict(row, col))), sizeof(pose4Predict(row, col)));
            }
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
        double meanTargetDimension = 3;
        double measurementDeviation = 0.1;
        eot_param para = {
            .accelerationDeviation = 0.1,
            .rotationalAccelerationDeviation = M_PI/360,
            .survivalProbability = 0.99,
            .meanBirths = 0.01,
            .measurementVariance = measurementDeviation*measurementDeviation,
            .meanMeasurements = 20,
            .meanClutter = 8,
            .priorVelocityCovariance = Eigen::DiagonalMatrix<double, 2>(0.01, 0.01),
            .priorTurningRateDeviation = M_PI/360,
            .meanTargetDimension = meanTargetDimension,
            .meanPriorExtent = meanTargetDimension * Eigen::Matrix2d::Identity(),
            .priorExtentDegreeFreedom = 1000,
            .degreeFreedomPrediction = 1000,
            .numParticles = 300,
            .ratioLegacyParticles = 0.2,
            .regularizationDeviation = meanTargetDimension/30,
            .detectionThreshold = 0.5,
            .thresholdPruning = 1e-2,
            .numOuterIterations = 2
        };
        EOT sim_eot;
        sim_eot.init(para);
        string path(argv[2]);
        vector<boost::filesystem::path> stream = streamFile(path);
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer("3D Viewer"));
        auto stramIterator = stream.begin();
        unordered_map<uint64_t, pair<size_t, PO>> trackID_count_PO_map;
        viewer->setCameraPosition(100, 0, 100, 0, 0, 1, 1, 0, 0);
        while (!viewer->wasStopped()) {
            vector<measurement> Measurements;
            vector<size_t> promissing_new_t_idx;
            vector<Eigen::Matrix2d> p_n_t_eigenvectors;
            vector<Eigen::Vector2d> p_n_t_extents;
            Eigen::Matrix4d pose4Predict;
            grid_para grid_parameters;
            double scanTime;
            uint64_t frame_idx;
            loadData(*stramIterator, Measurements, promissing_new_t_idx, p_n_t_eigenvectors, p_n_t_extents, pose4Predict, grid_parameters, scanTime, frame_idx);

            vector<PO> potential_objects_out;
            sim_eot.eot_track(Measurements, promissing_new_t_idx, p_n_t_eigenvectors, p_n_t_extents, pose4Predict, grid_parameters, scanTime, frame_idx, potential_objects_out);

            pcl::PointCloud<pcl::PointXYZ>::Ptr show_cloud_M(new pcl::PointCloud<pcl::PointXYZ>);
            double step(0.1);
            for(size_t m=0; m<Measurements.size(); ++m){
                show_cloud_M->push_back(pcl::PointXYZ(Measurements[m](1), Measurements[m](0), 0));
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr show_cloud_tracked_PO(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr show_cloud_untracked_PO(new pcl::PointCloud<pcl::PointXYZ>);
            // Tracked or untracked
            unordered_map<uint64_t, pair<size_t, PO>> tmp_trackID_count_PO_map;
            Eigen::Matrix2d rot_mat = pose4Predict.block<2, 2>(0, 0);
            if(potential_objects_out.size()>0){
                for(size_t t=0; t<potential_objects_out.size(); ++t){
                    pair<size_t, PO> tmp_count_PO = make_pair(1, potential_objects_out[t]);
                    uint64_t PO_label = potential_objects_out[t].label.label_v;
                    bool is_curr_PO_tracked = (trackID_count_PO_map.find(PO_label) != trackID_count_PO_map.end());
                    // if(is_curr_PO_tracked){
                    //     Eigen::Matrix2d legacyExtent = rot_mat * trackID_count_PO_map[PO_label].second.extent.e * rot_mat.transpose();
                    //     potential_objects_out[t].extent.e = (trackID_count_PO_map[PO_label].first * legacyExtent + potential_objects_out[t].extent.e)
                    //                                         /(trackID_count_PO_map[PO_label].first + 1);
                    //     Eigen::Vector4d legacy_center(trackID_count_PO_map[PO_label].second.kinematic.p1, trackID_count_PO_map[PO_label].second.kinematic.p2, 0, 1);
                    //     legacy_center = pose4Predict * legacy_center;
                    //     potential_objects_out[t].kinematic.p1 = (trackID_count_PO_map[PO_label].first * legacy_center(0) + potential_objects_out[t].kinematic.p1)
                    //                                             /(trackID_count_PO_map[PO_label].first + 1);
                    //     potential_objects_out[t].kinematic.p2 = (trackID_count_PO_map[PO_label].first * legacy_center(1) + potential_objects_out[t].kinematic.p2)
                    //                                             /(trackID_count_PO_map[PO_label].first + 1);
                    //     tmp_count_PO = make_pair(trackID_count_PO_map[PO_label].first + 1, potential_objects_out[t]);
                    // }
                    vector<Eigen::Vector2d> tmpPolygon;
                    utilities::extent2Polygon(potential_objects_out[t].kinematic, potential_objects_out[t].extent.eigenvalues, 
                                            potential_objects_out[t].extent.eigenvectors, 1.0, tmpPolygon);
                    tmpPolygon.push_back(tmpPolygon[0]);
                    for(size_t v=0; v<(tmpPolygon.size()-1); ++v){
                        Eigen::Vector2d edge = tmpPolygon[v+1] - tmpPolygon[v];
                        double edge_length = edge.norm();
                        for(double b=0; b<=edge_length; b+=step){
                            Eigen::Vector2d tmpP = tmpPolygon[v] + b*edge/edge_length;
                            if(is_curr_PO_tracked){
                                show_cloud_tracked_PO->push_back(pcl::PointXYZ(tmpP(1), tmpP(0), 0));
                            }else{
                                show_cloud_untracked_PO->push_back(pcl::PointXYZ(tmpP(1), tmpP(0), 0));
                            }
                        }
                    }
                    tmp_trackID_count_PO_map[PO_label] = tmp_count_PO;
                }
            }
            trackID_count_PO_map = tmp_trackID_count_PO_map;
            // Static or moving
            // for(size_t t=0; t<potential_objects_out.size(); ++t){
            //     double v1 = potential_objects_out[t].kinematic.v1;
            //     double v2 = potential_objects_out[t].kinematic.v2;
            //     bool is_curr_PO_static = (sqrt(v1*v1 + v2*v2) < 1);
            //     vector<Eigen::Vector2d> tmpPolygon;
            //     utilities::extent2Polygon(potential_objects_out[t].kinematic, potential_objects_out[t].extent.eigenvalues, 
            //                             potential_objects_out[t].extent.eigenvectors, 1.0, tmpPolygon);
            //     tmpPolygon.push_back(tmpPolygon[0]);
            //     for(size_t v=0; v<(tmpPolygon.size()-1); ++v){
            //         Eigen::Vector2d edge = tmpPolygon[v+1] - tmpPolygon[v];
            //         double edge_length = edge.norm();
            //         for(double b=0; b<=edge_length; b+=step){
            //             Eigen::Vector2d tmpP = tmpPolygon[v] + b*edge/edge_length;
            //             if(is_curr_PO_static){
            //                 show_cloud_tracked_PO->push_back(pcl::PointXYZ(tmpP(1), tmpP(0), 0));
            //             }else{
            //                 show_cloud_untracked_PO->push_back(pcl::PointXYZ(tmpP(1), tmpP(0), 0));
            //             }
            //         }
            //     }
            // }

            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color_cloud_M(show_cloud_M, 0, 0, 255);
            viewer->addPointCloud<pcl::PointXYZ>(show_cloud_M, single_color_cloud_M, "cloud_M");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_M");

            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color_cloud_tracked_PO(show_cloud_tracked_PO, 0, 255, 0);
            viewer->addPointCloud<pcl::PointXYZ>(show_cloud_tracked_PO, single_color_cloud_tracked_PO, "cloud_tracked_PO");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tracked_PO");

            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color_cloud_untracked_PO(show_cloud_untracked_PO, 255, 0, 0);
            viewer->addPointCloud<pcl::PointXYZ>(show_cloud_untracked_PO, single_color_cloud_untracked_PO, "cloud_untracked_PO");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_untracked_PO");

            stramIterator++;
            if(stramIterator == stream.end()){
                // stramIterator = stream.begin();
                break;
            }
            viewer->spinOnce (100);
            viewer->removeAllPointClouds();
            viewer->removeAllShapes();
        }
    }
    return 0;
}
#endif