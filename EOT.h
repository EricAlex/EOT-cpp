#pragma once
#include "utilities.h"
#include "globaldef.h"
#include "third_party/dbscan.h"
#include "third_party/easylogging++.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stack>
#include <set>
#include <string>
#include <limits>
#include <numeric>
#include <utility>
#include "omp.h"

#define DEBUG	false

using namespace std;
using namespace std::chrono;

class EOT{
    public:
        EOT(){
            el::Configurations conf("logger.conf");
            el::Loggers::reconfigureLogger("default", conf);
            el::Loggers::reconfigureAllLoggers(conf);
            m_defaultLogger_ = el::Loggers::getLogger("default");

            m_omp_numProcs_ = omp_get_num_procs()*3/4;
            m_omp_numProcs_ = m_omp_numProcs_>1?m_omp_numProcs_:2;
            omp_set_num_threads(m_omp_numProcs_);
        }

    	~EOT(){}

        void init(const eot_param& init_parameters){
            m_param_ = init_parameters;
            if((m_param_.ratioLegacyParticles<(1/double(m_param_.numParticles)))||(m_param_.ratioLegacyParticles>0.5)){
                m_defaultLogger_->error("Illegal ratioLegacyParticles: %v, should be between 0 and 0.5.", m_param_.ratioLegacyParticles);
                m_legacy_particles_mod_ = m_param_.numParticles;
            }else{
                m_legacy_particles_mod_ = 1/m_param_.ratioLegacyParticles;
            }
            m_current_max_label_num_ = 0;
        }

    	void eot_track(const vector<measurement>& measurements, 
                       const vector<size_t>& promissing_new_t_idx,
                       const vector<Eigen::Matrix2d>& p_n_t_eigenvectors,
                       const vector<Eigen::Vector2d>& p_n_t_extents,
                       const Eigen::Matrix4d& pose4Predict,
                       const grid_para& measurements_paras, 
                       const double delta_time, 
                       const uint64_t frame_idx, 
                       vector<PO>& potential_objects_out);
    private:
        void timer_start(){_time_start = steady_clock::now();}
        void timer_stop(){_time_finish = steady_clock::now();}
        float getTimer_millisec(){return (float)duration_cast<microseconds>(_time_finish - _time_start).count()/1000.0;}
        void update_grid_map_param(const grid_para& measurements_paras);
        void coord2index(const double p1, const double p2, uint32_t& index);
        void index2coord(const uint32_t index, double& p1, double& p2);
        void find_neighbors_(const uint32_t index, const uint32_t label, stack<uint32_t>& neighbors);
        void update_grid_label_(const uint32_t grid_index, const uint32_t label, stack<uint32_t>& neighbors);
        bool performPrediction(const double delta_time,
                               const vector<measurement>& measurements, 
                               const vector<size_t>& promissing_new_t_idx,
                               const vector<Eigen::Matrix2d>& p_n_t_eigenvectors,
                               const vector<Eigen::Vector2d>& p_n_t_extents,
                               const Eigen::Matrix4d& pose4Predict);
        void dataAssociationBP(const vector< Eigen::Vector2d >& inputDA, vector<double>& outputDA);
        void getWeightsUnknown(const vector< vector<double> >& logWeights_m_p, 
                               const double oldExistence, 
                               const int skipIndex, 
                               vector<double>& weights,
                               double& updatedExistence);
        void updateParticles(const vector< vector<double> >& logWeights_m_p,
                             const int target,
                             bool is_legacy_target);
        void resampleSystematic(const vector<double>& weights, vector<size_t>& indexes);
        bool getPromisingNewTargets(const vector<measurement>& ori_measurements, 
                                    const vector<size_t>& promissing_new_t_idx,
                                    const vector<Eigen::Matrix2d>& p_n_t_eigenvectors,
                                    const vector<Eigen::Vector2d>& p_n_t_extents,
                                    vector<size_t>& newIndexes, 
                                    vector<size_t>& selected_new_t_idx_idx, 
                                    vector<measurement>& ordered_measurements);
        void copyMat2Vec(const Eigen::Matrix2d& mat, vector<double>& vec);
        void copyVec2Mat(const vector<double>& vec, Eigen::Matrix2d& mat);
        void copyEvec2Vec(const Eigen::Vector2d& evec, vector<double>& vec);
        void copyVec2Evec(const vector<double>& vec, Eigen::Vector2d& evec);
        void maskMeasurements4LegacyPOs(const vector<measurement>& measurements,
                                        vector< vector<size_t> >& mask_m_t,
                                        vector< vector<size_t> >& mask_t_m);
	private:
        int m_omp_numProcs_;
        steady_clock::time_point _time_start;
        steady_clock::time_point _time_finish;
        el::Logger* m_defaultLogger_;
        grid_para m_grid_para_;
        double m_grid_resolution_reciprocal_;
        uint32_t m_rows_;
        uint32_t m_cols_;
        uint32_t m_cols_shift_;
        double m_grid_resolution_;
        eot_param m_param_;
        size_t m_legacy_particles_mod_;
        string m_err_str_;
        uint64_t m_current_max_label_num_;
        unordered_map<uint32_t, uint32_t> m_index_label_map_;
        vector<po_label> m_currentLabels_t_;
        vector< vector<po_kinematic> > m_currentParticlesKinematic_t_p_;
        vector<double> m_currentExistences_t_;
        vector< vector<po_extent> > m_currentParticlesExtent_t_p_;
        vector<bool> m_currentLegacyParticlesFlags_t_;
        vector<PO> m_currentPotentialObjects_t_;
        vector<PO> m_currentAugmentedPOs_t_;
        vector<size_t> m_currentMeanMeasurements_t_;
};