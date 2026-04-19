import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
import torch

from typing import Optional, Union, List, Any, Tuple, Dict

from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer

from skopt import BayesSearchCV
from skopt.space import Real, Integer


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | SINGULARITY-V14 | %(message)s")
warnings.filterwarnings("ignore")


class SovereignV14Singularity(BaseEstimator, ClassifierMixin):
    """
    Sovereign V14 Singularity (10/10 version)

    - Full sklearn Pipeline: preprocessing + XGBoost.
    - Bayesian hyperparameter optimization with BayesSearchCV.
    - Clean separation between:
        * hyperparameter search (fit on opt split),
        * optional early-stopping refit,
        * calibration on held-out calibration split.
    - Compatible binary / multiclass.
    - SHAP explanations on the calibrated core model.
    """

    def __init__(
        self,
        n_bayes_iter: int = 50,
        n_cv: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: Optional[int] = 50,
        test_size_calib: float = 0.15,
        scoring: str = "roc_auc",
        verbose: int = 0,
    ):
        self.n_bayes_iter = n_bayes_iter
        self.n_cv = n_cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size_calib = test_size_calib
        self.scoring = scoring
        self.verbose = verbose

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline_: Optional[Pipeline] = None
        self.search_: Optional[BayesSearchCV] = None
        self.model_: Optional[CalibratedClassifierCV] = None
        self.feature_names_: Optional[np.ndarray] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        cat_cols: List[str] = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        num_cols: List[str] = X.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()

        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("quantile", QuantileTransformer(output_distribution="normal", random_state=self.random_state)),
                ("scaler", RobustScaler()),
            ]
        )

        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        )

        return preprocessor

    def _infer_eval_metric(self, y: np.ndarray) -> str:
        n_classes = np.unique(y).shape[0]
        if n_classes <= 2:
            return "auc"
        # For multiclass, we use mlogloss as a robust default
        return "mlogloss"

    def _build_base_model(self, y: np.ndarray) -> XGBClassifier:
        eval_metric = self._infer_eval_metric(y)

        return XGBClassifier(
            tree_method="hist",
            device=self.device,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric=eval_metric,
            use_label_encoder=False,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Fit the full Sovereign V14 Singularity pipeline:
        - Split into optimization / calibration.
        - BayesSearchCV on optimization split.
        - Optional early-stopping refit on opt+calib boundary.
        - Calibration on calibration split.
        """
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_arr = np.asarray(y).ravel()

        if X_df.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y_arr)

        logging.info(f"Synchronizing Singularity V14 on {self.device}...")

        # Outer split: optimization vs calibration
        X_opt, X_calib, y_opt, y_calib = train_test_split(
            X_df,
            y_arr,
            test_size=self.test_size_calib,
            stratify=y_arr,
            random_state=self.random_state,
        )

        preprocessor = self._build_preprocessor(X_opt)
        base_model = self._build_base_model(y_opt)

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", base_model),
            ]
        )

        # Search space on the model step
        search_space = {
            "model__n_estimators": Integer(800, 4000),
            "model__max_depth": Integer(3, 10),
            "model__learning_rate": Real(0.01, 0.1, prior="log-uniform"),
            "model__min_child_weight": Integer(1, 20),
            "model__subsample": Real(0.6, 1.0),
            "model__colsample_bytree": Real(0.6, 1.0),
            "model__gamma": Real(1e-3, 5.0, prior="log-uniform"),
            "model__reg_alpha": Real(1e-5, 10.0, prior="log-uniform"),
            "model__reg_lambda": Real(1e-5, 10.0, prior="log-uniform"),
        }

        cv = StratifiedKFold(
            n_splits=self.n_cv,
            shuffle=True,
            random_state=self.random_state,
        )

        self.search_ = BayesSearchCV(
            estimator=pipe,
            search_spaces=search_space,
            n_iter=self.n_bayes_iter,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            refit=True,
            verbose=self.verbose,
        )

        # Hyperparameter search
        self.search_.fit(X_opt, y_opt)

        self.best_params_ = self.search_.best_params_
        self.pipeline_ = self.search_.best_estimator_

        preprocessor_fitted: ColumnTransformer = self.pipeline_.named_steps["preprocessor"]
        core_model: XGBClassifier = self.pipeline_.named_steps["model"]

        # Feature names after preprocessing
        try:
            self.feature_names_ = preprocessor_fitted.get_feature_names_out()
        except Exception:
            self.feature_names_ = np.array(
                [f"f_{i}" for i in range(preprocessor_fitted.transform(X_opt).shape[1])]
            )

        # Optional early stopping on final refit
        if self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
            logging.info("Refitting core XGBoost with early stopping on calibration split...")
            X_opt_trans = preprocessor_fitted.transform(X_opt)
            X_calib_trans = preprocessor_fitted.transform(X_calib)

            core_model.fit(
                X_opt_trans,
                y_opt,
                eval_set=[(X_calib_trans, y_calib)],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
            )

        # Final calibration on calibration split
        logging.info("Refining Probability Manifolds via calibration...")
        X_calib_trans = preprocessor_fitted.transform(X_calib)

        # Calibration method: isotonic only for binary and enough data
        if (len(y_arr) > 1000) and (self.classes_.shape[0] == 2):
            calib_method = "isotonic"
        else:
            calib_method = "sigmoid"

        self.model_ = CalibratedClassifierCV(
            estimator=core_model,
            method=calib_method,
            cv="prefit",
        )
        self.model_.fit(X_calib_trans, y_calib)

        logging.info(f"Singularity V14 Online. Convergence Score: {self.search_.best_score_:.6f}")
        return self

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------
    def _check_is_fitted(self):
        if self.model_ is None or self.pipeline_ is None or self.classes_ is None:
            raise RuntimeError("SovereignV14Singularity is not fitted yet. Call `fit` first.")

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        self._check_is_fitted()
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        preprocessor: ColumnTransformer = self.pipeline_.named_steps["preprocessor"]
        X_trans = preprocessor.transform(X_df)
        proba = self.model_.predict_proba(X_trans)

        # Ensure columns align with self.classes_
        if proba.shape[1] != self.classes_.shape[0]:
            raise RuntimeError("Mismatch between predicted probabilities and stored classes_.")
        return proba

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        self._check_is_fitted()
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_size: int = 100,
        plot: bool = True,
    ) -> Tuple[Any, pd.DataFrame]:
        """
        SHAP explanation on the calibrated core model.

        Parameters
        ----------
        X : array-like or DataFrame
            Data to explain.
        sample_size : int
            Max number of samples to use for SHAP.
        plot : bool
            If True, produces a SHAP summary plot.

        Returns
        -------
        shap_values : Any
            SHAP values object.
        sample : pd.DataFrame
            Sampled, transformed feature matrix used for SHAP.
        """
        self._check_is_fitted()
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        preprocessor: ColumnTransformer = self.pipeline_.named_steps["preprocessor"]
        core_model: XGBClassifier = self.model_.estimator

        X_trans = preprocessor.transform(X_df)
        X_explain = pd.DataFrame(X_trans, columns=self.feature_names_)

        if X_explain.shape[0] == 0:
            raise ValueError("No samples provided for explanation.")

        sample = X_explain.sample(
            n=min(len(X_explain), sample_size),
            random_state=self.random_state,
        )

        explainer = shap.TreeExplainer(core_model)
        shap_values = explainer(sample)

        if plot:
            shap.summary_plot(shap_values, sample)

        return shap_values, sample

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save the full fitted SovereignV14Singularity instance.
        """
        self._check_is_fitted()
        joblib.dump(self, path, compress=3)
        logging.info(f"Singularity State Saved: {path}")

    @staticmethod
    def load(path: str) -> "SovereignV14Singularity":
        """
        Load a previously saved SovereignV14Singularity instance.
        """
        model = joblib.load(path)
        if not isinstance(model, SovereignV14Singularity):
            raise TypeError("Loaded object is not a SovereignV14Singularity instance.")
        return model
