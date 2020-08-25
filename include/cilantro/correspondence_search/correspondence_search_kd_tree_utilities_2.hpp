#pragma once

#include <cilantro/core/common_pair_evaluators.hpp>
#include <iostream>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, typename TreeT, typename CorrSetT, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>>
    void findkNNCorrespondencesUnidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                             const TreeT &ref_tree,
                                             bool ref_is_first,
                                             CorrSetT &correspondences,
                                             typename EvaluatorT::OutputScalar max_distance,
                                             Eigen::MatrixXf &src_feat,
                                             Eigen::MatrixXf &dst_feat,
                                             const EvaluatorT &evaluator = EvaluatorT(),
                                             int k = 10,
                                             double max_feat_dist = 10.0
                                             )
    {
        using CorrIndexT = typename CorrSetT::value_type::Index;
        using CorrScalarT = typename CorrSetT::value_type::Scalar;        
        if (ref_tree.getPointsMatrixMap().cols() == 0) {
            correspondences.clear();
            return;
        }
        
        CorrSetT corr_tmp(query_pts.cols());
        std::vector<bool> keep(query_pts.cols());
        typename TreeT::NeighborhoodResult nn;
        typename EvaluatorT::OutputScalar dist;
        
        if (ref_is_first) {
#pragma omp parallel for shared(corr_tmp) private(nn, dist) schedule(dynamic, 256)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                ref_tree.kNNInRadiusSearch(query_pts.col(i), k, max_distance, nn);
                keep[i] = false;
                //corr_tmp[i].resize(k);
                int idx = 0;
                double min_dist_tmp = 1.0e6;
                for(auto k1 = 0;k1 < std::min<int>(k,nn.size());k1++)
                {
                    //keep[i] = (!nn.empty() && (dist = evaluator(nn[k1].index, i, nn[k1].value)) < max_distance) or keep[i];
                    if(!nn.empty() and (dist = evaluator(nn[k1].index, i, nn[k1].value)) < max_distance and dist < min_dist_tmp and (dst_feat.col(nn[k1].index)-src_feat.col(i)).norm() < max_feat_dist)
                
                    //if (keep[i])
                        {
                            corr_tmp[i] = {static_cast<CorrIndexT>(nn[k1].index), static_cast<CorrIndexT>(i), static_cast<CorrScalarT>(dist)};
                            keep[i] = true;
                        }
                 }
                //corr_tmp[i].resize(idx);
            }
        } else {
            //std::cout<<query_pts.cols()<<" "<<src_feat.cols()<<" "<<dst_feat.cols()<<"\n";
#pragma omp parallel for shared(corr_tmp) private(nn, dist) schedule(dynamic, 256)            
            for (size_t i = 0; i < query_pts.cols(); i++) {
                ref_tree.kNNInRadiusSearch(query_pts.col(i), k, max_distance, nn);
                keep[i] = false;
                double min_dist_tmp = 1.0e6;
                //corr_tmp[i].resize(k);
                int idx = 0;
                for(auto k1 = 0;k1 < std::min<int>(k,nn.size());k1++)
                {                    
                  //  keep[i] = (!nn.empty() && (dist = evaluator(i, nn[0].index, nn[0].value)) < max_distance) or keep[i];
                    if(!nn.empty() and (dist = evaluator(i, nn[k1].index, nn[k1].value)) < max_distance and dist < min_dist_tmp and (dst_feat.col(nn[k1].index)-src_feat.col(i)).norm() < max_feat_dist)                    
                    //if (keep[i])
                    {
                        corr_tmp[i] = {static_cast<CorrIndexT>(i), static_cast<CorrIndexT>(nn[k1].index), static_cast<CorrScalarT>(dist)};
                        keep[i] = true;
                    }
                }
                //corr_tmp[i].resize(idx);
            }
       
        }

        correspondences.resize(corr_tmp.size());
        size_t count = 0;
        for (size_t i = 0; i < corr_tmp.size(); i++) {
            if (keep[i]) correspondences[count++] = corr_tmp[i];
        }
        correspondences.resize(count);
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename TreeT, typename CorrSetT, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>>
    inline CorrSetT findkNNCorrespondencesUnidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                                        const TreeT &ref_tree,
                                                        bool ref_is_first,
                                                        typename EvaluatorT::OutputScalar max_distance,
                                                        Eigen::MatrixXf &src_feat,
                                                        Eigen::MatrixXf &dst_feat,
                                                        const EvaluatorT &evaluator = EvaluatorT(),
                                                        int k = 10, double max_feat_dist = 10.0)
    {
        CorrSetT corr_set;
        findkNNCorrespondencesUnidirectional<ScalarT,EigenDim>(query_pts, ref_tree, ref_is_first, corr_set, max_distance,src_feat,dst_feat, evaluator,k,max_feat_dist);
        return corr_set;
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename FirstTreeT, typename SecondTreeT, typename CorrSetT, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>>
    void findkNNCorrespondencesBidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &first_points,
                                            const ConstVectorSetMatrixMap<ScalarT,EigenDim> &second_points,
                                            const FirstTreeT &first_tree,
                                            const SecondTreeT &second_tree,
                                            CorrSetT &correspondences,
                                            typename EvaluatorT::OutputScalar max_distance,                                            
                                            Eigen::MatrixXf &src_feat,
                                            Eigen::MatrixXf &dst_feat,
                                            bool require_reciprocal = false,
                                            const EvaluatorT &evaluator = EvaluatorT(),
                                            int k = 10,
                                            double max_feat_dist = 10.0)
    {
        CorrSetT corr_first_to_second, corr_second_to_first;
        findkNNCorrespondencesUnidirectional<ScalarT,EigenDim>(first_points, second_tree, false, corr_first_to_second, max_distance, dst_feat, src_feat, evaluator,k,max_feat_dist);
        findkNNCorrespondencesUnidirectional<ScalarT,EigenDim>(second_points, first_tree, true, corr_second_to_first, max_distance, src_feat, dst_feat, evaluator,k,max_feat_dist);

        typename CorrSetT::value_type::IndicesLexicographicalComparator comparator;

#pragma omp parallel sections
        {
#pragma omp section
            std::sort(corr_first_to_second.begin(), corr_first_to_second.end(), comparator);
#pragma omp section
            std::sort(corr_second_to_first.begin(), corr_second_to_first.end(), comparator);
        }

        correspondences.clear();
        correspondences.reserve(corr_first_to_second.size() + corr_second_to_first.size());

        if (require_reciprocal) {
            std::set_intersection(corr_first_to_second.begin(), corr_first_to_second.end(),
                                  corr_second_to_first.begin(), corr_second_to_first.end(),
                                  std::back_inserter(correspondences), comparator);
        } else {
            std::set_union(corr_first_to_second.begin(), corr_first_to_second.end(),
                           corr_second_to_first.begin(), corr_second_to_first.end(),
                           std::back_inserter(correspondences), comparator);
        }
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename FirstTreeT, typename SecondTreeT, typename CorrSetT, class EvaluatorT = DistanceEvaluator<ScalarT,ScalarT>>
    inline CorrSetT findkNNCorrespondencesBidirectional(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &first_points,
                                                       const ConstVectorSetMatrixMap<ScalarT,EigenDim> &second_points,
                                                       const FirstTreeT &first_tree,
                                                       const SecondTreeT &second_tree,
                                                       typename EvaluatorT::OutputScalar max_distance,
                                                       Eigen::MatrixXf &src_feat,
                                                       Eigen::MatrixXf &dst_feat,                                                       
                                                       bool require_reciprocal = false,
                                                       const EvaluatorT &evaluator = EvaluatorT(),
                                                       int k = 10,
                                                       double max_feat_dist = 10.0)
    {
        CorrSetT corr_set;
        findkNNCorrespondencesBidirectional<ScalarT,EigenDim>(first_points, second_points, first_tree, second_tree, corr_set, max_distance,src_feat, dst_feat, require_reciprocal, evaluator,k,max_feat_dist);
        return corr_set;
    }
}
