"""Edge prediction model based on CatBoost."""
from typing import Dict, Any, Optional, List, Union
from catboost import CatBoostClassifier
from embiggen.edge_prediction.sklearn_like_edge_prediction_adapter import (
    SklearnLikeEdgePredictionAdapter,
)


class CatBoostEdgePrediction(SklearnLikeEdgePredictionAdapter):
    """Edge prediction model based on CatBoost."""

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        l2_leaf_reg: float = 3.0,
        model_size_reg: float = 0.5,
        rsm: float = 1.0,
        loss_function: Optional[str] = None,
        border_count: int = 254,
        feature_border_type: str = "GreedyLogSum",
        per_float_feature_quantization: Optional[List[str]] = None,
        input_borders: Optional[str] = None,
        output_borders: Optional[str] = None,
        fold_permutation_block: int = 1,
        od_pval: Optional[float] = None,
        od_wait: Optional[int] = None,
        od_type: Optional[str] = None,
        nan_mode: str = "Min",
        counter_calc_method: str = "SkipTest",
        leaf_estimation_iterations: int = 1,
        leaf_estimation_method: str = "Newton",
        thread_count: int = -1,
        use_best_model: Optional[bool] = None,
        best_model_min_trees: int = 1,
        verbose: bool = False,
        metric_period: int = 1,
        ctr_leaf_count_limit: int = 16,
        store_all_simple_ctr: bool = False,
        max_ctr_complexity: int = 4,
        has_time: bool = False,
        allow_const_label: bool = False,
        target_border=None,
        classes_count: Optional[int] = None,
        class_weights=None,
        auto_class_weights: str = "Balanced",
        class_names=None,
        one_hot_max_size: Optional[int] = None,
        random_strength: float = 1.0,
        name: str = "experiment",
        ignored_features=None,
        train_dir: Optional[str] = None,
        custom_metric=None,
        eval_metric=None,
        bagging_temperature: int = 1,
        save_snapshot=None,
        snapshot_file=None,
        snapshot_interval: int = 600,
        fold_len_multiplier=None,
        used_ram_limit=None,
        gpu_ram_part: float = 0.95,
        pinned_memory_size=None,
        allow_writing_files=None,
        final_ctr_computation_mode: str = "Default",
        approx_on_full_history: bool = False,
        boosting_type=None,
        simple_ctr=None,
        combinations_ctr: Optional[List[str]] = None,
        per_feature_ctr: Optional[List[str]] = None,
        ctr_target_border_count: int = 1,
        task_type: Optional[str] = None,
        devices=None,
        bootstrap_type: str = "MVS",
        subsample: float = 1.0,
        mvs_reg=None,
        sampling_unit: str = "Object",
        sampling_frequency: str = "PerTree",
        dev_score_calc_obj_block_size: int = 5000000,
        dev_efb_max_buckets: int = 1024,
        sparse_features_conflict_fraction: float = 0.0,
        early_stopping_rounds=None,
        cat_features=None,
        grow_policy: str = "SymmetricTree",
        min_data_in_leaf: int = 1,
        max_leaves: int = 31,
        score_function: str = "Cosine",
        leaf_estimation_backtracking=None,
        monotone_constraints=None,
        feature_weights=None,
        penalties_coefficient: float = 1.0,
        first_feature_use_penalties=None,
        per_object_feature_penalties=None,
        model_shrink_rate: float = 0.0,
        model_shrink_mode=None,
        langevin=None,
        diffusion_temperature: float = 0.0,
        posterior_sampling=None,
        boost_from_average=None,
        text_features=None,
        tokenizers=None,
        dictionaries=None,
        feature_calcers=None,
        text_processing=None,
        embedding_features=None,
        callback=None,
        eval_fraction=None,
        edge_embedding_methods: Union[List[str], str] = "Concatenate",
        training_unbalance_rate: float = 1.0,
        use_edge_metrics: bool = False,
        use_scale_free_distribution: bool = True,
        prediction_batch_size: int = 2**12,
        random_state: int = 42,
    ):
        """Build a CatBoost Edge prediction model."""
        self._kwargs = dict(
            iterations=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            l2_leaf_reg=l2_leaf_reg,
            model_size_reg=model_size_reg,
            rsm=rsm,
            loss_function=loss_function,
            border_count=border_count,
            feature_border_type=feature_border_type,
            per_float_feature_quantization=per_float_feature_quantization,
            input_borders=input_borders,
            output_borders=output_borders,
            fold_permutation_block=fold_permutation_block,
            od_pval=od_pval,
            od_wait=od_wait,
            od_type=od_type,
            nan_mode=nan_mode,
            counter_calc_method=counter_calc_method,
            leaf_estimation_iterations=leaf_estimation_iterations,
            leaf_estimation_method=leaf_estimation_method,
            thread_count=thread_count,
            use_best_model=use_best_model,
            best_model_min_trees=best_model_min_trees,
            verbose=verbose,
            metric_period=metric_period,
            ctr_leaf_count_limit=ctr_leaf_count_limit,
            store_all_simple_ctr=store_all_simple_ctr,
            max_ctr_complexity=max_ctr_complexity,
            has_time=has_time,
            allow_const_label=allow_const_label,
            target_border=target_border,
            classes_count=classes_count,
            class_weights=class_weights,
            auto_class_weights=auto_class_weights,
            class_names=class_names,
            one_hot_max_size=one_hot_max_size,
            random_strength=random_strength,
            name=name,
            ignored_features=ignored_features,
            train_dir=train_dir,
            custom_metric=custom_metric,
            eval_metric=eval_metric,
            bagging_temperature=bagging_temperature,
            save_snapshot=save_snapshot,
            snapshot_file=snapshot_file,
            snapshot_interval=snapshot_interval,
            fold_len_multiplier=fold_len_multiplier,
            used_ram_limit=used_ram_limit,
            gpu_ram_part=gpu_ram_part,
            pinned_memory_size=pinned_memory_size,
            allow_writing_files=allow_writing_files,
            final_ctr_computation_mode=final_ctr_computation_mode,
            approx_on_full_history=approx_on_full_history,
            boosting_type=boosting_type,
            simple_ctr=simple_ctr,
            combinations_ctr=combinations_ctr,
            per_feature_ctr=per_feature_ctr,
            ctr_target_border_count=ctr_target_border_count,
            task_type=task_type,
            devices=devices,
            bootstrap_type=bootstrap_type,
            subsample=subsample,
            mvs_reg=mvs_reg,
            sampling_unit=sampling_unit,
            sampling_frequency=sampling_frequency,
            dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
            dev_efb_max_buckets=dev_efb_max_buckets,
            sparse_features_conflict_fraction=sparse_features_conflict_fraction,
            early_stopping_rounds=early_stopping_rounds,
            cat_features=cat_features,
            grow_policy=grow_policy,
            min_data_in_leaf=min_data_in_leaf,
            max_leaves=max_leaves,
            score_function=score_function,
            leaf_estimation_backtracking=leaf_estimation_backtracking,
            monotone_constraints=monotone_constraints,
            feature_weights=feature_weights,
            penalties_coefficient=penalties_coefficient,
            first_feature_use_penalties=first_feature_use_penalties,
            per_object_feature_penalties=per_object_feature_penalties,
            model_shrink_rate=model_shrink_rate,
            model_shrink_mode=model_shrink_mode,
            langevin=langevin,
            diffusion_temperature=diffusion_temperature,
            posterior_sampling=posterior_sampling,
            boost_from_average=boost_from_average,
            text_features=text_features,
            tokenizers=tokenizers,
            dictionaries=dictionaries,
            feature_calcers=feature_calcers,
            text_processing=text_processing,
            embedding_features=embedding_features,
            callback=callback,
            eval_fraction=eval_fraction,
        )

        super().__init__(
            model_instance=CatBoostClassifier(
                **self._kwargs,
                random_state=random_state,
            ),
            edge_embedding_methods=edge_embedding_methods,
            training_unbalance_rate=training_unbalance_rate,
            use_edge_metrics=use_edge_metrics,
            use_scale_free_distribution=use_scale_free_distribution,
            prediction_batch_size=prediction_batch_size,
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        return dict(
            iterations=2,
            max_depth=2,
        )

    def parameters(self) -> Dict[str, Any]:
        return dict(
            **super().parameters(),
            **self._kwargs,
        )

    @classmethod
    def model_name(cls) -> str:
        """Return the name of the model."""
        return "CatBoost"

    @classmethod
    def library_name(cls) -> str:
        """Return name of the model."""
        return "CatBoost"