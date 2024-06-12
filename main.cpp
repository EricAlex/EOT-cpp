#include <iostream>
#include <sstream>
#include <unistd.h>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "globaldef.h"
#include "utilities.h"
#include "EOT.h"
#include "third_party/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

using namespace std;

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
            .numParticles = 400,
            .ratioLegacyParticles = 0.2,
            .regularizationDeviation = meanTargetDimension/30,
            .detectionThreshold = 0.5,
            .thresholdPruning = 1e-3,
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

            for(size_t t=0; t<potential_objects_out.size(); ++t){
                double pos1 = potential_objects_out[t].kinematic.p2 - 0.5*potential_objects_out[t].extent.eigenvalues(1);
                double pos2 = potential_objects_out[t].kinematic.p1 + 0.5*potential_objects_out[t].extent.eigenvalues(0);
                pcl::PointXYZ text_position(pos1, pos2, 0.0);
                // string text_content = to_string(potential_objects_out[t].label.label_v);
                double speed = sqrt(potential_objects_out[t].kinematic.v1*potential_objects_out[t].kinematic.v1
                                    +potential_objects_out[t].kinematic.v2*potential_objects_out[t].kinematic.v2);
                stringstream stream;
                stream << std::fixed << std::setprecision(2) << speed;
                string text_content = stream.str();

                viewer->addText3D(text_content, text_position, 1, 1.0, 1.0, 1.0, "trackID_"+to_string(t));
            }

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