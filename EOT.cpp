#include "EOT.h"

EOT::EOT(const eot_param& init_parameters) {
    m_param_ = init_parameters;
}

EOT::~EOT() {}

void EOT::update_grid_map_param(const grid_para& measurements_paras){
    m_grid_para_ = measurements_paras;
    m_grid_resolution_reciprocal_ = 1 / m_grid_para_.grid_res;
    m_cols_ = static_cast<uint32_t>((measurements_paras.dim1_max - measurements_paras.dim1_min) / m_grid_para_.grid_res);
    m_rows_ = static_cast<uint32_t>((measurements_paras.dim2_max - measurements_paras.dim2_min) / m_grid_para_.grid_res);
    uint32_t power = log2(m_cols_);
    if (m_cols_ == (1U << power)) {
        m_cols_shift_ = power;
    } else {
        m_cols_shift_ = power + 1;
        m_cols_ = (1U << m_cols_shift_);
    }
}

void EOT::coord2index(const double p1, const double p2, uint32_t& index){
    uint32_t c_idx = (p1 - m_grid_para_.dim1_min) * m_grid_resolution_reciprocal_ + 1;
    uint32_t r_idx = (p2 - m_grid_para_.dim2_min) * m_grid_resolution_reciprocal_ + 1;
    index = (r_idx << (m_cols_shift_)) + c_idx;
}

void EOT::find_neighbors_(const uint32_t index, const uint32_t label, stack<uint32_t>& neighbors) {
    set<uint32_t> neighbor_index;
    neighbor_index.insert(index - m_cols_); // down neighbor
    neighbor_index.insert(index - m_cols_ - 1); // down left neighbor
    neighbor_index.insert(index - m_cols_ + 1); // down right neighbor
    neighbor_index.insert(index + m_cols_); // up neighbor
    neighbor_index.insert(index + m_cols_ - 1); // up left neighbor
    neighbor_index.insert(index + m_cols_ + 1); // up right neighbor
    neighbor_index.insert(index - 1); // left neighbor
    neighbor_index.insert(index + 1); // right neighbor
    auto iter = neighbor_index.begin();
    while (iter != neighbor_index.end()) {
        update_grid_label_(*iter, label, neighbors);
        iter++;
    }
}

void EOT::update_grid_label_(const uint32_t grid_index, const uint32_t label, stack<uint32_t>& neighbors) {
  if ((m_index_label_map_.find(grid_index) != m_index_label_map_.end())) {
    if (m_index_label_map_[grid_index] == INIT_VAILD_GRID_LABEL) {
      neighbors.push(grid_index);
      m_index_label_map_[grid_index] = label;
    }
  }
}

bool EOT::performPrediction(const double delta_time){
    if((m_currentParticlesKinematic_t_p_.size()==m_currentParticlesExtent_t_p_.size())
        &&(m_currentParticlesExtent_t_p_.size()==m_currentExistences_t_.size())){
        for(size_t t=0; t<m_currentExistences_t_.size(); ++t){
            m_currentExistences_t_[t] *= m_param_.survivalProbability;
            #pragma omp parallel for
            for(size_t p=0; p<m_param_.numParticles; ++p){
                // // prediction with rotation
                // double rot_ang = m_currentParticlesKinematic_t_p_[t][p].t * delta_time;
                // Eigen::Matrix2d rot_mat;
                // rot_mat << cos(rot_ang), sin(rot_ang),
                //            -sin(rot_ang), cos(rot_ang);
                // Eigen::MatrixXd S = (rot_mat * m_currentParticlesExtent_t_p_[t][p].e * rot_mat.transpose()) / m_param_.degreeFreedomPrediction;
                Eigen::MatrixXd S = m_currentParticlesExtent_t_p_[t][p].e / m_param_.degreeFreedomPrediction;
                m_currentParticlesExtent_t_p_[t][p].e = utilities::sampleWishart(m_param_.degreeFreedomPrediction, S);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(m_currentParticlesExtent_t_p_[t][p].e);
                m_currentParticlesExtent_t_p_[t][p].eigenvalues = eigensolver.eigenvalues();
                m_currentParticlesExtent_t_p_[t][p].eigenvectors = eigensolver.eigenvectors();
            }
            #pragma omp parallel for
            for(size_t p=0; p<m_param_.numParticles; ++p){
                double r1 = utilities::sampleGaussian(0, m_param_.accelerationDeviation);
                double r2 = utilities::sampleGaussian(0, m_param_.accelerationDeviation);
                m_currentParticlesKinematic_t_p_[t][p].p1 += (m_currentParticlesKinematic_t_p_[t][p].v1*delta_time
                                                             + 0.5*delta_time*delta_time*r1);
                m_currentParticlesKinematic_t_p_[t][p].p2 += (m_currentParticlesKinematic_t_p_[t][p].v2*delta_time
                                                             + 0.5*delta_time*delta_time*r2);
                m_currentParticlesKinematic_t_p_[t][p].v1 += delta_time*r1;
                m_currentParticlesKinematic_t_p_[t][p].v2 += delta_time*r2;
                m_currentParticlesKinematic_t_p_[t][p].t += delta_time*utilities::sampleGaussian(0, m_param_.rotationalAccelerationDeviation);
                // // predict mean number of measurments with shape extention
                // m_currentParticlesKinematic_t_p_[t][p].s = utilities::mean_number_of_measurements(m_currentParticlesExtent_t_p_[t][p].eigenvalues, m_grid_para_.grid_res);
                // assign default mean number of measurments
                m_currentParticlesKinematic_t_p_[t][p].s = m_param_.meanMeasurements;
            }
        }
        for(size_t t=0; t<m_currentPotentialObjects_t_.size(); ++t){
            m_currentPotentialObjects_t_[t].kinematic.p1 += delta_time*m_currentPotentialObjects_t_[t].kinematic.v1;
            m_currentPotentialObjects_t_[t].kinematic.p2 += delta_time*m_currentPotentialObjects_t_[t].kinematic.v2;
            // // prediction with rotation
            // double rot_ang = m_currentPotentialObjects_t_[t].kinematic.t * delta_time;
            // Eigen::Matrix2d rot_mat;
            // rot_mat << cos(rot_ang), sin(rot_ang),
            //            -sin(rot_ang), cos(rot_ang);
            // m_currentPotentialObjects_t_[t].extent.e = rot_mat * m_currentPotentialObjects_t_[t].extent.e * rot_mat.transpose();
            // m_currentPotentialObjects_t_[t].extent.eigenvectors = rot_mat * m_currentPotentialObjects_t_[t].extent.eigenvectors * rot_mat.transpose();
        }
    }else{
        m_err_str_ = "[ERROR] performPrediction: legacy target numbers do not match " + to_string(m_currentParticlesKinematic_t_p_.size())
                    + "/" + to_string(m_currentParticlesExtent_t_p_.size()) + "/" + to_string(m_currentExistences_t_.size());
        return false;
    }
    return true;
}

double dist_func(const vec2d& t1, const vec2d& t2){
    return sqrt(std::pow(t1[0]-t2[0], 2) + std::pow(t1[1]-t2[1], 2));
}

bool EOT::getPromisingNewTargets(const vector<measurement>& measurements, 
                                 vector<size_t>& newIndexes, 
                                 vector<measurement>& ordered_measurements){
    
    // // grow seed grid
    // uint32_t label = INIT_VAILD_GRID_LABEL + m_predicted_boxes_in_range_.size();
    // auto it = m_index_label_map_.begin();
    // while (it != m_index_label_map_.end()) {
    //     if (it->second == INIT_VAILD_GRID_LABEL) {
    //     vector<uint32_t> temp_indices;
    //     ++label;
    //     it->second = label;
    //     stack<uint32_t>
    //         neighbors;  // normal road test max neighbors deep about 600
    //     neighbors.emplace(it->first);
    //     while (!neighbors.empty()) {
    //         uint32_t cur_index = neighbors.top();
    //         temp_indices.push_back(cur_index);
    //         neighbors.pop();
    //         find_neighbors_(cur_index, label, neighbors);
    //     }
    //     label_cluster_indices.push_back(temp_indices);
    //     }
    //     it++;
    // }

    unordered_set<size_t> remainIndexes, rmedLegacyIndexes;
    for(size_t m=0; m<measurements.size(); ++m){
        remainIndexes.insert(m);
    }
    double sigmaRatio(1);
    vector< vector<Eigen::Vector2d> > legacyPOPolygons;
    for(size_t t=0; t<m_currentPotentialObjects_t_.size(); ++t){
        vector<Eigen::Vector2d> tmpPolygon;
        utilities::extent2Polygon(m_currentPotentialObjects_t_[t].kinematic, m_currentPotentialObjects_t_[t].extent.eigenvalues, 
                                  m_currentPotentialObjects_t_[t].extent.eigenvectors, sigmaRatio, tmpPolygon);
        legacyPOPolygons.push_back(tmpPolygon);
    }
    if(m_currentPotentialObjects_t_.size()>0){
        for(size_t m=0; m<measurements.size(); ++m){
            measurement M = measurements[m];
            for(size_t t=0; t<m_currentPotentialObjects_t_.size(); ++t){
                double radius = sigmaRatio*sqrt(std::pow(m_currentPotentialObjects_t_[t].extent.eigenvalues(1), 2) + std::pow(m_currentPotentialObjects_t_[t].extent.eigenvalues(0), 2));
                if((M(0)<(m_currentPotentialObjects_t_[t].kinematic.p1-radius))||(M(0)>(m_currentPotentialObjects_t_[t].kinematic.p1+radius))
                    ||(M(1)<(m_currentPotentialObjects_t_[t].kinematic.p2-radius))||(M(1)>(m_currentPotentialObjects_t_[t].kinematic.p2+radius))){
                    continue;
                }else{
                    if(utilities::isInPolygon(legacyPOPolygons[t], M)){
                        remainIndexes.erase(remainIndexes.find(m));
                        rmedLegacyIndexes.insert(m);
                        break;
                    }
                }
            }
        }
    }

    vector<vec2d> m4Cluster;
    for(auto it=remainIndexes.begin(); it!=remainIndexes.end(); it++){
        m4Cluster.push_back(vec2d{measurements[*it](0), measurements[*it](1)});
    }
    auto dbscan = DBSCAN<vec2d, double>();
    dbscan.Run(&m4Cluster, 2, m_param_.meanTargetDimension, 1, &dist_func);
    auto noise = dbscan.Noise;
    auto clusters = dbscan.Clusters;
    for(auto& n:noise){
        ordered_measurements.push_back(measurement(m4Cluster[n][0], m4Cluster[n][1]));
    }
    for(size_t i=0; i<clusters.size(); i++){
        measurement m_centor(0, 0);
        for(size_t j=0; j<clusters[i].size(); j++){
            m_centor += measurement(m4Cluster[clusters[i][j]][0], m4Cluster[clusters[i][j]][1]);
        }
        m_centor = m_centor/clusters[i].size();
        size_t centor_idx(0);
        double min_dist(numeric_limits<double>::infinity());
        for(size_t j=0; j<clusters[i].size(); j++){
            measurement dist_vec = measurement(m4Cluster[clusters[i][j]][0], m4Cluster[clusters[i][j]][1]) - m_centor;
            double dist = dist_vec.norm();
            if(dist<min_dist){
                min_dist = dist;
                centor_idx = j;
            }
        }
        for(size_t j=0; j<clusters[i].size(); j++){
            if(j!=centor_idx){
                ordered_measurements.push_back(measurement(m4Cluster[clusters[i][j]][0], m4Cluster[clusters[i][j]][1]));
            }
        }
        ordered_measurements.push_back(measurement(m4Cluster[clusters[i][centor_idx]][0], m4Cluster[clusters[i][centor_idx]][1]));
        newIndexes.push_back(ordered_measurements.size()-1);
    }
    for(auto it=rmedLegacyIndexes.begin(); it!=rmedLegacyIndexes.end(); it++){
        ordered_measurements.push_back(measurements[*it]);
    }

    return true;
}

void EOT::dataAssociationBP(const vector< Eigen::Vector2d >& inputDA, vector<double>& outputDA){
    // perform DA
    size_t numDA = inputDA.size();
    outputDA.resize(numDA);
    double sumInputDA(1.0);
    vector<double> tempDA(numDA);
    size_t max_val_idx(0);
    for(size_t i=0; i<numDA; ++i){
        tempDA[i] = inputDA[i](1)/inputDA[i](0);
        sumInputDA += tempDA[i];
        if(tempDA[i] > tempDA[max_val_idx]){
            max_val_idx = i;
        }
    }
    bool has_nan(false);
    for(size_t i=0; i<numDA; ++i){
        outputDA[i] = 1.0 / (sumInputDA - tempDA[i]);
        has_nan = has_nan||isnan(outputDA[i]);
    }
    // make hard DA decision in case outputDA involves NANs
    if(has_nan){
        for(size_t i=0; i<numDA; ++i){
            outputDA[i] = 0.0;
        }
        outputDA[max_val_idx] = 1.0;
    }
}

void EOT::getWeightsUnknown(const vector< vector<double> >& logWeights_m_p, 
                            const double oldExistence, 
                            const int skipIndex, 
                            vector<double>& weights,
                            double& updatedExistence){
    int numMeasurements = logWeights_m_p.size();
    vector<double> tmpWeights_p(m_param_.numParticles);
    vector<double> exp_tmpWeights_p(m_param_.numParticles);
    size_t max_val_idx(0);
    for(size_t p=0; p<m_param_.numParticles; ++p){
        double tmpSum(0);
        for(int m=0; m<numMeasurements; ++m){
            if(m != skipIndex){
                tmpSum += logWeights_m_p[m][p];
            }
        }
        tmpWeights_p[p] = tmpSum;
        exp_tmpWeights_p[p] = exp(tmpWeights_p[p]);
        if(tmpWeights_p[p] > tmpWeights_p[max_val_idx]){
            max_val_idx = p;
        }
    }
    double aliveUpdate = std::accumulate(std::begin(exp_tmpWeights_p), std::end(exp_tmpWeights_p), 0.0)/exp_tmpWeights_p.size();
    if(isinf(aliveUpdate)){
        updatedExistence = 1;
    }else{
        double alive = oldExistence * aliveUpdate;
        double dead = (1 - oldExistence);
        updatedExistence = alive / (dead + alive);
    }
    double max_tmpWeights_val(0.0);
    if(tmpWeights_p.size()>0){
        max_tmpWeights_val = tmpWeights_p[max_val_idx];
    }
    double tmpSum(0);
    for(size_t p=0; p<m_param_.numParticles; ++p){
        tmpWeights_p[p] = exp(tmpWeights_p[p] - max_tmpWeights_val);
        tmpSum += tmpWeights_p[p];
    }
    for(size_t p=0; p<m_param_.numParticles; ++p){
        weights[p] = tmpWeights_p[p]/tmpSum;
    }
}

void EOT::resampleSystematic(const vector<double>& weights, vector<size_t>& indexes){
    indexes.resize(m_param_.numParticles);
    vector<double> cumWeights(m_param_.numParticles);
    double tmpSum(0);
    for(size_t p=0; p<m_param_.numParticles; ++p){
        tmpSum += weights[p];
        cumWeights[p] = tmpSum;
    }
    vector<double> grid(m_param_.numParticles+1);
    double rd_add = utilities::sampleUniform(0, 1)/double(m_param_.numParticles);
    double step = (1-1/double(m_param_.numParticles))/double(m_param_.numParticles-1);
    #pragma omp parallel for
    for(size_t p=0; p<m_param_.numParticles; ++p){
        grid[p] = p * step + rd_add;
    }
    grid[m_param_.numParticles] = 1.0;
    size_t i(0), j(0);
    while(i<m_param_.numParticles){
        if(grid[i] < cumWeights[j]){
            indexes[i] = j;
            i++;
        }else{
            j++;
        }
    }
}

void EOT::copyStruct(po_extent& dest, const po_extent& src){
    dest.e = src.e;
    dest.eigenvalues = src.eigenvalues;
    dest.eigenvectors = src.eigenvectors;
}

void EOT::updateParticles(const vector< vector<double> >& logWeights_m_p, const int target){
    int numMeasurements = logWeights_m_p.size();
    vector<double> tmpWeights_p(m_param_.numParticles);
    vector<double> exp_tmpWeights_p(m_param_.numParticles);
    size_t max_val_idx(0);
    for(size_t p=0; p<m_param_.numParticles; ++p){
        double tmpSum(0);
        for(int m=0; m<numMeasurements; ++m){
            tmpSum += logWeights_m_p[m][p];
        }
        tmpWeights_p[p] = tmpSum;
        exp_tmpWeights_p[p] = exp(tmpWeights_p[p]);
        if(tmpWeights_p[p] > tmpWeights_p[max_val_idx]){
            max_val_idx = p;
        }
    }
    double aliveUpdate = std::accumulate(std::begin(exp_tmpWeights_p), std::end(exp_tmpWeights_p), 0.0)/exp_tmpWeights_p.size();
    if(isinf(aliveUpdate)){
        m_currentExistences_t_[target] = 1;
    }else{
        double alive = m_currentExistences_t_[target] * aliveUpdate;
        double dead = 1 - m_currentExistences_t_[target];
        m_currentExistences_t_[target] = alive/(dead+alive);
    }
    if(m_currentExistences_t_[target] != 0){
        double max_tmpWeights_val(0.0);
        if(tmpWeights_p.size()>0){
            max_tmpWeights_val = tmpWeights_p[max_val_idx];
        }
        double tmpSum(0);
        for(size_t p=0; p<m_param_.numParticles; ++p){
            tmpWeights_p[p] = exp(tmpWeights_p[p] - max_tmpWeights_val);
            tmpSum += tmpWeights_p[p];
        }
        for(size_t p=0; p<m_param_.numParticles; ++p){
            tmpWeights_p[p] = tmpWeights_p[p]/tmpSum;
        }
        vector<size_t> indexes(m_param_.numParticles);
        resampleSystematic(tmpWeights_p, indexes);
        vector<po_kinematic> tmpKinematic_p(m_param_.numParticles);
        vector<po_extent> tmpExtent_p(m_param_.numParticles);
        #pragma omp parallel for
        for(size_t p=0; p<m_param_.numParticles; ++p){
            tmpKinematic_p[p] = m_currentParticlesKinematic_t_p_[target][p];
            copyStruct(tmpExtent_p[p], m_currentParticlesExtent_t_p_[target][p]);
        }
        #pragma omp parallel for
        for(size_t p=0; p<m_param_.numParticles; ++p){
            m_currentParticlesKinematic_t_p_[target][p] = tmpKinematic_p[indexes[p]];
            copyStruct(m_currentParticlesExtent_t_p_[target][p], tmpExtent_p[p]);
            m_currentParticlesKinematic_t_p_[target][p].p1 += m_param_.regularizationDeviation * utilities::sampleGaussian(0, 1);
            m_currentParticlesKinematic_t_p_[target][p].p2 += m_param_.regularizationDeviation * utilities::sampleGaussian(0, 1);
        }
    }else{
        double nanValue = nan("");
        #pragma omp parallel for
        for(size_t p=0; p<m_param_.numParticles; ++p){
            m_currentParticlesKinematic_t_p_[target][p].p1 = nanValue;
            m_currentParticlesKinematic_t_p_[target][p].p2 = nanValue;
            m_currentParticlesKinematic_t_p_[target][p].v1 = nanValue;
            m_currentParticlesKinematic_t_p_[target][p].v2 = nanValue;
            m_currentParticlesKinematic_t_p_[target][p].t = nanValue;
            m_currentParticlesKinematic_t_p_[target][p].s = nanValue;
            m_currentParticlesExtent_t_p_[target][p].e = (Eigen::Matrix2d() << nanValue, nanValue, nanValue, nanValue).finished();
            m_currentParticlesExtent_t_p_[target][p].eigenvalues = (Eigen::Vector2d() << nanValue, nanValue).finished();
            m_currentParticlesExtent_t_p_[target][p].eigenvectors = (Eigen::Matrix2d() << nanValue, nanValue, nanValue, nanValue).finished();
        }
    }
}

void EOT::eot_track(const vector<measurement>& ori_measurements, 
                    const grid_para& measurements_paras, 
                    const double delta_time, 
                    const uint64_t frame_idx, 
                    vector<PO>& potential_objects_out){
    timer_start();
    // init configuration parameters
    Eigen::Matrix2d totalCovariance = m_param_.meanPriorExtent*m_param_.meanPriorExtent;
    totalCovariance.array() += m_param_.measurementVariance;
    double areaSize = (measurements_paras.dim1_max-measurements_paras.dim1_min)*(measurements_paras.dim2_max-measurements_paras.dim2_min);
    double uniformWeight = log(1/areaSize);
    Eigen::MatrixXd priorExtentShape = m_param_.meanPriorExtent*(m_param_.priorExtentDegreeFreedom-m_param_.meanPriorExtent.cols()-1);
    double nanValue = nan("");
    double constantFactor = areaSize*(double(m_param_.meanMeasurements)/m_param_.meanClutter);
    double gate_ratio(2.0);
    double measurementSD = sqrt(m_param_.measurementVariance);

    // init measurement grid parameters
    update_grid_map_param(measurements_paras);
    size_t numMeasurements = ori_measurements.size();

    // perform prediction step
    if(!performPrediction(delta_time)){
        cout<<m_err_str_<<endl;
        return;
    }
    double exp_minus_meanMeasurements = exp(-double(m_param_.meanMeasurements));
    for(size_t t=0; t<m_currentExistences_t_.size(); ++t){
        double currentAlive = m_currentExistences_t_[t]*exp_minus_meanMeasurements;
        double currentDead = 1 - m_currentExistences_t_[t];
        m_currentExistences_t_[t] = currentAlive/(currentDead+currentAlive);
    }
    size_t numTargets(m_currentParticlesKinematic_t_p_.size());
    int numLegacy = numTargets;

    // get indexes of promising new objects
    vector<size_t> newIndexes; // store indexes in reverse order
    vector<measurement> measurements;
    getPromisingNewTargets(ori_measurements, newIndexes, measurements);
    size_t numNew = newIndexes.size();
    unordered_map<size_t, bool> newIndexes_map;
    for(size_t i=0; i<numNew; ++i){
        po_label label_n{frame_idx, newIndexes[i]};
        m_currentLabels_t_.push_back(label_n);
        newIndexes_map[newIndexes[i]] = true;
    }

    // initialize belief propagation (BP) message passing
    double init_new_existence = m_param_.meanBirths * exp_minus_meanMeasurements/(m_param_.meanBirths * exp_minus_meanMeasurements +1);
    vector< vector<double> > newWeights_t_p(numNew, vector<double>(m_param_.numParticles));
    Eigen::Matrix2d proposalCovariance = 2 * totalCovariance;
    Eigen::Matrix2d proposalCovariance_sqrt = utilities::sqrtm(proposalCovariance);
    for(size_t t=0; t<numNew; ++t){
        Eigen::Vector2d proposalMean = measurements[newIndexes[t]];
        vector<po_kinematic> kinematic_p(m_param_.numParticles);
        vector<po_extent> extent_p(m_param_.numParticles);
        #pragma omp parallel for
        for(size_t p=0; p<m_param_.numParticles; ++p){
            Eigen::Vector2d posi = proposalMean + proposalCovariance_sqrt * Eigen::Vector2d(utilities::sampleGaussian(0, 1), utilities::sampleGaussian(0, 1));
            kinematic_p[p].p1 = posi(0);
            kinematic_p[p].p2 = posi(1);
            newWeights_t_p[t][p] = uniformWeight - log(utilities::mvnormPDF(posi, proposalMean, proposalCovariance));
            extent_p[p].e = utilities::sampleInverseWishart(m_param_.priorExtentDegreeFreedom, priorExtentShape);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(extent_p[p].e);
            extent_p[p].eigenvalues = eigensolver.eigenvalues();
            extent_p[p].eigenvectors = eigensolver.eigenvectors();
        }
        m_currentParticlesKinematic_t_p_.push_back(kinematic_p);
        m_currentParticlesExtent_t_p_.push_back(extent_p);
        m_currentExistences_t_.push_back(init_new_existence);
    }
    vector< vector<double> > currentExistencesExtrinsic_m_t(numMeasurements, m_currentExistences_t_);
    vector< vector< vector<double> > > weightsExtrinsic_t_m_p(numLegacy, vector< vector<double> >(numMeasurements, vector<double>(m_param_.numParticles, nanValue)));
    vector< vector< vector<double> > > weightsExtrinsicNew_t_m_p(numNew, vector< vector<double> >(numMeasurements, vector<double>(m_param_.numParticles, nanValue)));
    vector< vector< vector<double> > > likelihood1_t_m_p(numLegacy, vector< vector<double> >(numMeasurements, vector<double>(m_param_.numParticles, 0.0)));
    vector< vector< vector<double> > > likelihoodNew1_t_m_p(numNew, vector< vector<double> >(numMeasurements, vector<double>(m_param_.numParticles, nanValue)));

    for(size_t outer=0; outer<m_param_.numOuterIterations; ++outer){
        // perform one BP message passing iteration for each measurement
        unordered_map<uint32_t, vector<double>> outputDA;
        unordered_map<uint32_t, unordered_map<size_t, size_t> > targetIndexes;
        for(int m=(numMeasurements-1); m>=0; --m){
            vector< Eigen::Vector2d > inputDA(numLegacy, Eigen::Vector2d(0.0, 0.0));
            for(size_t t=0; t<numLegacy; ++t){
                if(outer == 0){
                    #pragma omp parallel for
                    for(size_t p=0; p<m_param_.numParticles; ++p){
                        likelihood1_t_m_p[t][m][p] = constantFactor * utilities::measurement_likelihood_(m_currentParticlesKinematic_t_p_[t][p], 
                            m_currentParticlesExtent_t_p_[t][p].eigenvalues, m_currentParticlesExtent_t_p_[t][p].eigenvectors, gate_ratio, 
                            measurements[m], measurementSD);
                    }
                    inputDA[t](1) = currentExistencesExtrinsic_m_t[m][t] 
                        * (std::accumulate(std::begin(likelihood1_t_m_p[t][m]), std::end(likelihood1_t_m_p[t][m]), 0.0)/likelihood1_t_m_p[t][m].size());
                }else{
                    double product_sum(0.0);
                    for(size_t p=0; p<m_param_.numParticles; ++p){
                        product_sum += weightsExtrinsic_t_m_p[t][m][p] * likelihood1_t_m_p[t][m][p];
                    }
                    inputDA[t](1) = currentExistencesExtrinsic_m_t[m][t] * product_sum;
                }
                inputDA[t](0) = 1.0;
            }
            int targetIndex = numLegacy - 1;
            unordered_map<size_t, size_t> targetIndexesCurrent;
            // only new targets with index >= measurement index are connected to measurement
            for(int t=(numMeasurements-1); t>=m; --t){
                if(newIndexes_map.find(t) != newIndexes_map.end()){
                    targetIndex = targetIndex + 1;
                    targetIndexesCurrent[t] = targetIndex;
                    Eigen::Vector2d tmp_inputDA;
                    if(outer == 0){
                        vector<double> weights_p(m_param_.numParticles);
                        double temp_sum(0.0);
                        for(size_t p=0; p<m_param_.numParticles; ++p){
                            weights_p[p] = exp(newWeights_t_p[targetIndex-numLegacy][p]);
                            temp_sum += weights_p[p];
                        }
                        double product_sum(0.0);
                        for(size_t p=0; p<m_param_.numParticles; ++p){
                            weights_p[p] /= temp_sum;
                            likelihoodNew1_t_m_p[targetIndex-numLegacy][m][p] = constantFactor * utilities::measurement_likelihood_(m_currentParticlesKinematic_t_p_[targetIndex][p],
                                m_currentParticlesExtent_t_p_[targetIndex][p].eigenvalues, m_currentParticlesExtent_t_p_[targetIndex][p].eigenvectors, gate_ratio,
                                measurements[m], measurementSD);
                            product_sum += weights_p[p] * likelihoodNew1_t_m_p[targetIndex-numLegacy][m][p];
                        }
                        tmp_inputDA(1) = currentExistencesExtrinsic_m_t[m][targetIndex] * product_sum;
                    }else{
                        double product_sum(0.0);
                        for(size_t p=0; p<m_param_.numParticles; ++p){
                            product_sum += weightsExtrinsicNew_t_m_p[targetIndex-numLegacy][m][p] * likelihoodNew1_t_m_p[targetIndex-numLegacy][m][p];
                        }
                        tmp_inputDA(1) = currentExistencesExtrinsic_m_t[m][targetIndex] * product_sum;
                    }
                    tmp_inputDA(0) = 1.0;
                    if(t == m){
                        tmp_inputDA(0) = 1.0 - currentExistencesExtrinsic_m_t[m][targetIndex];
                    }
                    inputDA.push_back(tmp_inputDA);
                }
            }
            targetIndexes[m] = targetIndexesCurrent;
            vector<double> tempOutDA;
            dataAssociationBP(inputDA, tempOutDA);
            outputDA[m] = tempOutDA;
        }

        // perform update step for legacy targets
        for(int t=0; t<numLegacy; ++t){
            vector< vector<double> > weights_m_p(numMeasurements, vector<double>(m_param_.numParticles, 0.0));
            for(int m=0; m<numMeasurements; ++m){
                double outputTmpDA = outputDA[m][t];
                #pragma omp parallel for
                for(size_t p=0; p<m_param_.numParticles; ++p){
                    weights_m_p[m][p] = log(1 + likelihood1_t_m_p[t][m][p]*outputTmpDA);
                }
            }
            // calculate extrinsic information for legacy targets (at all except last iteration) and belief (at last iteration)
            if(outer != (m_param_.numOuterIterations-1)){
                #pragma omp parallel for
                for(int m=0; m<numMeasurements; ++m){
                    getWeightsUnknown(weights_m_p, m_currentExistences_t_[t], m, weightsExtrinsic_t_m_p[t][m], currentExistencesExtrinsic_m_t[m][t]);
                }
            }else{
                updateParticles(weights_m_p, t);
            }
        }

        // perform update step for new targets
        int targetIndex = numLegacy - 1;
        for(int t=(numMeasurements-1); t>=0; --t){
            if(newIndexes_map.find(t) != newIndexes_map.end()){
                targetIndex = targetIndex + 1;
                vector< vector<double> > weights_m_p(numMeasurements+1, vector<double>(m_param_.numParticles, 0.0));
                #pragma omp parallel for
                for(size_t p=0; p<m_param_.numParticles; ++p){
                    weights_m_p[numMeasurements][p] = newWeights_t_p[targetIndex-numLegacy][p];
                }
                for(int m=0; m<=t; ++m){
                    double outputTmpDA = outputDA[m][targetIndexes[m][t]];
                    #pragma omp parallel for
                    for(size_t p=0; p<m_param_.numParticles; ++p){
                        double currentWeights = likelihoodNew1_t_m_p[targetIndex-numLegacy][m][p];
                        if(!isinf(outputTmpDA)){
                            currentWeights *= outputTmpDA;
                        }
                        if(m != t){
                            currentWeights += 1;
                        }
                        weights_m_p[m][p] = log(currentWeights);
                    }
                }

                // calculate extrinsic information for new targets (at all except last iteration) or belief (at last iteration)
                if(outer != (m_param_.numOuterIterations-1)){
                    #pragma omp parallel for
                    for(int m=0; m<=t; ++m){
                        getWeightsUnknown(weights_m_p, m_currentExistences_t_[targetIndex], m, 
                            weightsExtrinsicNew_t_m_p[targetIndex-numLegacy][m], currentExistencesExtrinsic_m_t[m][targetIndex]);
                    }
                }else{
                    updateParticles(weights_m_p, targetIndex);
                    #pragma omp parallel for
                    for(size_t p=0; p<m_param_.numParticles; ++p){
                        Eigen::Vector2d tmpSpeed = utilities::sampleMvNormal(Eigen::Vector2d(0.0, 0.0), m_param_.priorVelocityCovariance);
                        m_currentParticlesKinematic_t_p_[targetIndex][p].v1 = tmpSpeed(0);
                        m_currentParticlesKinematic_t_p_[targetIndex][p].v2 = tmpSpeed(1);
                        m_currentParticlesKinematic_t_p_[targetIndex][p].t = utilities::sampleGaussian(0, m_param_.priorTurningRateDeviation);
                    }
                }
            }
        }
    }

    #if DEBUG
        for(size_t t=0; t<m_currentExistences_t_.size(); ++t){
            cout<<"\t"<<m_currentExistences_t_[t];
        }
        cout<<endl;
    #endif

    // perform pruning
    for(size_t t=0; t<m_currentExistences_t_.size();){
        if(m_currentExistences_t_[t] < m_param_.thresholdPruning){
            m_currentExistences_t_.erase(m_currentExistences_t_.begin()+t);
            m_currentParticlesKinematic_t_p_.erase(m_currentParticlesKinematic_t_p_.begin()+t);
            m_currentParticlesExtent_t_p_.erase(m_currentParticlesExtent_t_p_.begin()+t);
            m_currentLabels_t_.erase(m_currentLabels_t_.begin()+t);
        }else{
            t++;
        }
    }

    // perform estimation
    m_currentPotentialObjects_t_.resize(0);
    for(size_t t=0; t<m_currentExistences_t_.size(); ++t){
        if(m_currentExistences_t_[t] > m_param_.detectionThreshold){
            PO tmpDetectedPO;
            tmpDetectedPO.label = m_currentLabels_t_[t];
            po_kinematic sum_kinematic = {0, 0, 0, 0, 0, 0};
            Eigen::Matrix2d sum_e = (Eigen::Matrix2d() << 0, 0, 0, 0).finished();
            for(size_t p=0; p<m_param_.numParticles; ++p){
                sum_kinematic.p1 += m_currentParticlesKinematic_t_p_[t][p].p1;
                sum_kinematic.p2 += m_currentParticlesKinematic_t_p_[t][p].p2;
                sum_kinematic.v1 += m_currentParticlesKinematic_t_p_[t][p].v1;
                sum_kinematic.v2 += m_currentParticlesKinematic_t_p_[t][p].v2;
                sum_kinematic.t += m_currentParticlesKinematic_t_p_[t][p].t;
                sum_kinematic.s += m_currentParticlesKinematic_t_p_[t][p].s;
                sum_e += m_currentParticlesExtent_t_p_[t][p].e;
            }
            tmpDetectedPO.kinematic.p1 = sum_kinematic.p1/double(m_param_.numParticles);
            tmpDetectedPO.kinematic.p2 = sum_kinematic.p2/double(m_param_.numParticles);
            tmpDetectedPO.kinematic.v1 = sum_kinematic.v1/double(m_param_.numParticles);
            tmpDetectedPO.kinematic.v2 = sum_kinematic.v2/double(m_param_.numParticles);
            tmpDetectedPO.kinematic.t = sum_kinematic.t/double(m_param_.numParticles);
            tmpDetectedPO.kinematic.s = sum_kinematic.s/double(m_param_.numParticles);
            tmpDetectedPO.extent.e = sum_e/double(m_param_.numParticles);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(tmpDetectedPO.extent.e);
            tmpDetectedPO.extent.eigenvalues = eigensolver.eigenvalues();
            tmpDetectedPO.extent.eigenvectors = eigensolver.eigenvectors();
            m_currentPotentialObjects_t_.push_back(tmpDetectedPO);
            potential_objects_out.push_back(tmpDetectedPO);
        }
    }

    timer_stop();
    cout<<"EOT done, costs "<<getTimer_millisec()<<" ms."<<endl;
}