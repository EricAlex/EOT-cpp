#pragma once
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
#include "omp.h"

#include "globaldef.h"
#include "utilities.h"
#include "dbscan.h"

using namespace std;
using namespace std::chrono;

class EOT{
    public:
        EOT(){}

    	~EOT(){}

        void init(const eot_param& init_parameters){m_param_ = init_parameters;}

    	void eot_track(const vector<measurement>& measurements, 
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
        void find_neighbors_(const uint32_t index, const uint32_t label, stack<uint32_t>& neighbors);
        void update_grid_label_(const uint32_t grid_index, const uint32_t label, stack<uint32_t>& neighbors);
        bool performPrediction(const double delta_time);
        void dataAssociationBP(const vector< Eigen::Vector2d >& inputDA, vector<double>& outputDA);
        void getWeightsUnknown(const vector< vector<double> >& logWeights_m_p, 
                               const double oldExistence, 
                               const int skipIndex, 
                               vector<double>& weights,
                               double& updatedExistence);
        void updateParticles(const vector< vector<double> >& logWeights_m_p, const int target);
        void resampleSystematic(const vector<double>& weights, vector<size_t>& indexes);
        bool getPromisingNewTargets(const vector<measurement>& ori_measurements, 
                                    vector<size_t>& newIndexes, 
                                    vector<measurement>& ordered_measurements);
        void copyStruct(po_extent& dest, const po_extent& src);
	private:
        steady_clock::time_point _time_start;
        steady_clock::time_point _time_finish;
        grid_para m_grid_para_;
        double m_grid_resolution_reciprocal_;
        uint32_t m_rows_;
        uint32_t m_cols_;
        uint32_t m_cols_shift_;
        double m_grid_resolution_;
        eot_param m_param_;
        string m_err_str_;
        unordered_map<uint32_t, uint32_t> m_index_label_map_;
        vector<po_label> m_currentLabels_t_;
        vector< vector<po_kinematic> > m_currentParticlesKinematic_t_p_;
        vector<double> m_currentExistences_t_;
        vector< vector<po_extent> > m_currentParticlesExtent_t_p_;
        vector<PO> m_currentPotentialObjects_t_;
};