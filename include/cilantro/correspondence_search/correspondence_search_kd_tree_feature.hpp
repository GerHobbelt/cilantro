#pragma once

#include <memory>
#include <cilantro/core/correspondence.hpp>
#include <cilantro/core/kd_tree.hpp>
//#include <cilantro/correspondence_search/correspondence_search_kd_tree_utilities.hpp>
#include <cilantro/correspondence_search/correspondence_search_kd_tree_utilities_2.hpp>
#include <iostream>

namespace cilantro {
//    template <typename T, typename = int>
//    struct IsIsometry : std::false_type {};
//
//    template <typename T>
//    struct IsIsometry<T, decltype((void) T::Mode, 0)> : std::conditional<T::Mode == Eigen::Isometry, std::true_type, std::false_type>::type {};

    template <class SearchFeatureAdaptorT, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class EvaluationFeatureAdaptorT = SearchFeatureAdaptorT, class EvaluatorT = DistanceEvaluator<typename SearchFeatureAdaptorT::Scalar,typename EvaluationFeatureAdaptorT::Scalar>, typename IndexT = size_t>
    class CorrespondenceSearchKDTreeFeature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef EvaluatorT Evaluator;

        typedef typename EvaluatorT::OutputScalar CorrespondenceScalar;

        typedef IndexT CorrespondenceIndex;

        typedef CorrespondenceSet<CorrespondenceScalar,CorrespondenceIndex> SearchResult;

        typedef typename SearchFeatureAdaptorT::Scalar SearchFeatureScalar;

        typedef KDTree<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension,DistAdaptor,CorrespondenceIndex> SearchTree;

        template <class EvalFeatAdaptorT = EvaluationFeatureAdaptorT, class = typename std::enable_if<std::is_same<EvalFeatAdaptorT,SearchFeatureAdaptorT>::value>::type>
        CorrespondenceSearchKDTreeFeature(SearchFeatureAdaptorT &dst_features,
                                   SearchFeatureAdaptorT &src_features, Eigen::MatrixXf &dst_feat, Eigen::MatrixXf &src_feat,
                                   EvaluatorT &evaluator, int k, double max_feat_dist)
                : dst_search_features_adaptor_(dst_features), src_search_features_adaptor_(src_features),
                  src_evaluation_features_adaptor_(src_features), evaluator_(evaluator),
                  search_dir_(CorrespondenceSearchDirection::SECOND_TO_FIRST),
                  max_distance_((CorrespondenceScalar)(0.01*0.01)),
                  inlier_fraction_(1.0), require_reciprocality_(false), one_to_one_(false), k_(k),max_feat_dist_(max_feat_dist)
        { 
             dst_feat_.resize(dst_feat.rows(),dst_feat.cols()); 
             dst_feat_ << dst_feat;
             src_feat_.resize(src_feat.rows(),src_feat.cols()); 
             src_feat_ << src_feat;
        }

        CorrespondenceSearchKDTreeFeature(SearchFeatureAdaptorT &dst_search_features,
                                   SearchFeatureAdaptorT &src_search_features,
                                   EvaluationFeatureAdaptorT &src_eval_features,
                                   EvaluatorT &evaluator)
                : dst_search_features_adaptor_(dst_search_features), src_search_features_adaptor_(src_search_features),
                  src_evaluation_features_adaptor_(src_eval_features), evaluator_(evaluator),
                  search_dir_(CorrespondenceSearchDirection::SECOND_TO_FIRST),
                  max_distance_((CorrespondenceScalar)(0.01*0.01)),
                  inlier_fraction_(1.0), require_reciprocality_(false), one_to_one_(false)
        {}

        CorrespondenceSearchKDTreeFeature& findCorrespondences() {
            switch (search_dir_) {
                case CorrespondenceSearchDirection::FIRST_TO_SECOND: {
                    if (!src_tree_ptr_) src_tree_ptr_.reset(new SearchTree(src_search_features_adaptor_.getFeaturesMatrixMap()));
                    findkNNCorrespondencesUnidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(dst_search_features_adaptor_.getFeaturesMatrixMap(), *src_tree_ptr_, false, correspondences_, max_distance_, evaluator_);
                    break;
                }
                case CorrespondenceSearchDirection::SECOND_TO_FIRST: {
                    if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_search_features_adaptor_.getFeaturesMatrixMap()));
                    findkNNCorrespondencesUnidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(src_search_features_adaptor_.getFeaturesMatrixMap(), *dst_tree_ptr_, true, correspondences_, max_distance_, evaluator_);
                    break;
                }
                case CorrespondenceSearchDirection::BOTH: {
                    if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_search_features_adaptor_.getFeaturesMatrixMap()));
                    if (!src_tree_ptr_) src_tree_ptr_.reset(new SearchTree(src_search_features_adaptor_.getFeaturesMatrixMap()));
                    findNNCorrespondencesBidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(dst_search_features_adaptor_.getFeaturesMatrixMap(), src_search_features_adaptor_.getFeaturesMatrixMap(), *dst_tree_ptr_, *src_tree_ptr_, correspondences_, max_distance_, require_reciprocality_, evaluator_);
                    break;
                }
            }

            filterCorrespondencesFraction(correspondences_, inlier_fraction_);
            if (one_to_one_)
                filterCorrespondencesOneToOne(correspondences_, search_dir_);

            return *this;
        }

        // Interface for ICP use
        template <class TransformT>
        CorrespondenceSearchKDTreeFeature& findCorrespondences(const TransformT &tform) {                        
//            if (IsIsometry<TransformT>::value && std::is_same<SearchTree,KDTree<FeatureScalar,FeatureAdaptorT::FeatureDimension,KDTreeDistanceAdaptors::L2>>::value) {
//                // Avoid re-building tree for src if transformation is rigid and metric is L2
//                switch (search_dir_) {
//                    case CorrespondenceSearchDirection::FIRST_TO_SECOND: {
//                        if (!src_tree_ptr_) src_tree_ptr_.reset(new SearchTree(src_features_adaptor_.getFeaturesMatrixMap()));
//                        findNNCorrespondencesUnidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(dst_features_adaptor_.transformFeatures(tform.inverse()).getTransformedFeaturesMatrixMap(), *src_tree_ptr_, false, correspondences_, max_distance_, evaluator_);
//                        break;
//                    }
//                    case CorrespondenceSearchDirection::SECOND_TO_FIRST: {
//                        if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_features_adaptor_.getFeaturesMatrixMap()));
//                        findNNCorrespondencesUnidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(src_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap(), *dst_tree_ptr_, true, correspondences_, max_distance_, evaluator_);
//                        break;
//                    }
//                    case CorrespondenceSearchDirection::BOTH: {
//                        if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_features_adaptor_.getFeaturesMatrixMap()));
//                        if (!src_tree_ptr_) src_tree_ptr_.reset(new SearchTree(src_features_adaptor_.getFeaturesMatrixMap()));
//                        findNNCorrespondencesBidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(dst_features_adaptor_.transformFeatures(tform.inverse()).getTransformedFeaturesMatrixMap(), src_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap(), *dst_tree_ptr_, *src_tree_ptr_, correspondences_, max_distance_, require_reciprocality_, evaluator_);
//                        break;
//                    }
//                }
//            } else {
//                // General case
//                switch (search_dir_) {
//                    case CorrespondenceSearchDirection::FIRST_TO_SECOND: {
//                        src_trans_tree_ptr_.reset(new SearchTree(src_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap()));
//                        findNNCorrespondencesUnidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(dst_features_adaptor_.getFeaturesMatrixMap(), *src_trans_tree_ptr_, false, correspondences_, max_distance_, evaluator_);
//                        break;
//                    }
//                    case CorrespondenceSearchDirection::SECOND_TO_FIRST: {
//                        if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_features_adaptor_.getFeaturesMatrixMap()));
//                        findNNCorrespondencesUnidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(src_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap(), *dst_tree_ptr_, true, correspondences_, max_distance_, evaluator_);
//                        break;
//                    }
//                    case CorrespondenceSearchDirection::BOTH: {
//                        if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_features_adaptor_.getFeaturesMatrixMap()));
//                        src_trans_tree_ptr_.reset(new SearchTree(src_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap()));
//                        findNNCorrespondencesBidirectional<FeatureScalar,FeatureAdaptorT::FeatureDimension,DistAdaptor,EvaluatorT>(dst_features_adaptor_.getFeaturesMatrixMap(), src_features_adaptor_.getTransformedFeaturesMatrixMap(), *dst_tree_ptr_, *src_trans_tree_ptr_, correspondences_, max_distance_, require_reciprocality_, evaluator_);
//                        break;
//                    }
//                }
//            }
            
            //std::vector<SearchResult> corresp_tmp;

            if (!std::is_same<SearchFeatureAdaptorT,EvaluationFeatureAdaptorT>::value ||
                &src_search_features_adaptor_ != (SearchFeatureAdaptorT *)(&src_evaluation_features_adaptor_))
            {
                src_evaluation_features_adaptor_.transformFeatures(tform);
            }

            switch (search_dir_) {
                case CorrespondenceSearchDirection::FIRST_TO_SECOND: {
                    src_trans_tree_ptr_.reset(new SearchTree(src_search_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap()));
                    findkNNCorrespondencesUnidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(dst_search_features_adaptor_.getFeaturesMatrixMap(), 
                        *src_trans_tree_ptr_, false, correspondences_, max_distance_,dst_feat_,src_feat_, evaluator_,k_,max_feat_dist_);
                    break;
                }
                case CorrespondenceSearchDirection::SECOND_TO_FIRST: {
                    if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_search_features_adaptor_.getFeaturesMatrixMap()));
                    findkNNCorrespondencesUnidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(src_search_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap(), 
                        *dst_tree_ptr_, true, correspondences_, max_distance_,src_feat_,dst_feat_, evaluator_,k_,max_feat_dist_);
                    break;
                }
                case CorrespondenceSearchDirection::BOTH: {
                    if (!dst_tree_ptr_) dst_tree_ptr_.reset(new SearchTree(dst_search_features_adaptor_.getFeaturesMatrixMap()));
                    src_trans_tree_ptr_.reset(new SearchTree(src_search_features_adaptor_.transformFeatures(tform).getTransformedFeaturesMatrixMap()));
                    findkNNCorrespondencesBidirectional<SearchFeatureScalar,SearchFeatureAdaptorT::FeatureDimension>(dst_search_features_adaptor_.getFeaturesMatrixMap(), 
                        src_search_features_adaptor_.getTransformedFeaturesMatrixMap(), *dst_tree_ptr_, *src_trans_tree_ptr_, correspondences_, max_distance_,
                         src_feat_,dst_feat_,require_reciprocality_, evaluator_,k_,max_feat_dist_);
                    break;
                }
            }

            //filter correspondances based on features
            
//             int count_set = 0;
//             correspondences_.resize(corresp_tmp.size());            
//             SearchResult final_correspondences(corresp_tmp.size());
//             std::vector<bool> has_correspondance(corresp_tmp.size());
// #pragma omp parallel for
//             for(auto k1 = 0;k1<corresp_tmp.size();k1++)
//             {
//                 has_correspondance[k1] = false;
//                 int min_idx = -1;
//                 double min_tmp_dist = 1.0e6;
//                 for(auto k2 = 0;k2 < corresp_tmp[k1].size();k2++)
//                 {
//                     if(corresp_tmp[k1][k2].value < min_tmp_dist and (dst_feat_.col(corresp_tmp[k1][k2].indexInFirst) - src_feat_.col(corresp_tmp[k1][k2].indexInSecond)).norm() < max_feat_dist_)
//                     {
//                         min_idx = k2;
//                         min_tmp_dist = corresp_tmp[k1][k2].value;
//                     }
                    
//                 }                

//                 if(min_idx >= 0)
//                 {
//                     final_correspondences[k1] = corresp_tmp[k1][min_idx];                     
//                     has_correspondance[k1] = true;
//                 }
//                 //std::cout<<" "<<k1;
//             }

//             for(auto k1 = 0;k1<corresp_tmp.size();k1++)
//             {
//                 if(has_correspondance[k1])
//                     correspondences_[count_set++] = final_correspondences[k1];
//             }

//             correspondences_.resize(count_set);


            //std::cout<<"correspondances "<<correspondences_.size()<<"\n";
            filterCorrespondencesFraction(correspondences_, inlier_fraction_);
            if (one_to_one_)
                filterCorrespondencesOneToOne(correspondences_, search_dir_);

            return *this;
        }

        inline const SearchResult& getCorrespondences() const { return correspondences_; }

        inline Evaluator& evaluator() { return evaluator_; }

        inline const CorrespondenceSearchDirection& getSearchDirection() const { return search_dir_; }

        inline CorrespondenceSearchKDTreeFeature& setSearchDirection(const CorrespondenceSearchDirection &search_dir) {
            search_dir_ = search_dir;
            return *this;
        }

        inline CorrespondenceScalar getMaxDistance() const { return max_distance_; }

        inline CorrespondenceSearchKDTreeFeature& setMaxDistance(CorrespondenceScalar dist_thresh) {
            max_distance_ = dist_thresh;
            return *this;
        }

        inline double getInlierFraction() const { return inlier_fraction_; }

        inline CorrespondenceSearchKDTreeFeature& setInlierFraction(double fraction) {
            inlier_fraction_ = fraction;
            return *this;
        }

        inline bool getRequireReciprocality() const { return require_reciprocality_; }

        inline CorrespondenceSearchKDTreeFeature& setRequireReciprocality(bool require_reciprocal) {
            require_reciprocality_ = require_reciprocal;
            return *this;
        }

        inline bool getOneToOne() const { return one_to_one_;  }

        inline CorrespondenceSearchKDTreeFeature& setOneToOne(bool one_to_one) {
            one_to_one_ = one_to_one;
            return *this;
        }

       private:
        SearchFeatureAdaptorT& dst_search_features_adaptor_;
        SearchFeatureAdaptorT& src_search_features_adaptor_;
        Eigen::MatrixXf dst_feat_, src_feat_;        

        EvaluationFeatureAdaptorT& src_evaluation_features_adaptor_;
        Evaluator& evaluator_;

        std::shared_ptr<SearchTree> dst_tree_ptr_;
        std::shared_ptr<SearchTree> src_tree_ptr_;
        std::shared_ptr<SearchTree> src_trans_tree_ptr_;

        CorrespondenceSearchDirection search_dir_;
        CorrespondenceScalar max_distance_;
        int k_ = 10;
        double max_feat_dist_ = 10;
        double inlier_fraction_;
        bool require_reciprocality_;
        bool one_to_one_;

        SearchResult correspondences_;
    };
}
