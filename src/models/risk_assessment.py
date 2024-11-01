#models/risk_assessment.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

class DengueRiskAssessment:
    """
    A class for assessing arbovirus fever risk across different cities using
    weather and temporal features.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the risk assessment model with input data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing required columns:
            - city: City identifier
            - date: Date of observation
            - arbovirus_bool: Binary indicator of arbovirus occurrence
            - Various weather and temporal features
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        required_columns = ['city', 'date', 'arbovirus_bool']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Initialize attributes with None
        self.city_risk_scores = None
        self.feature_importances = None
        
    
    def _select_meaningful_features(self, target_col='arbovirus_bool', corr_threshold=0.85):
        """
        Select meaningful features using correlation analysis and data types.
        
        Parameters
        ----------
        target_col : str, optional
            Target variable column name
        corr_threshold : float, optional
            Correlation threshold for feature elimination
            
        Returns
        -------
        List[str]
            List of selected feature names
        """
        # Exclude non-feature columns
        exclude_cols = [target_col, 'date', 'city']
        
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("No numeric features found for analysis")
        
        # Remove highly correlated features
        corr_matrix = self.df[feature_cols].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [
            column for column in upper_tri.columns 
            if any(upper_tri[column] > corr_threshold)
        ]
        
        feature_cols = [col for col in feature_cols if col not in to_drop]
        
        if not feature_cols:
            raise ValueError("No features remained after correlation analysis")
            
        return feature_cols
   
   
    def calculate_feature_importance(self, target_col='arbovirus_bool', n_features=10):
        """
        Calculate feature importance using mutual information and random forest.
        
        Parameters
        ----------
        target_col : str, optional
            Target variable column name
        n_features : int, optional
            Number of top features to return
            
        Returns
        -------
        Dict
            Dictionary containing feature importance metrics
        """
        # Use self to access class methods and attributes
        feature_cols = self._select_meaningful_features(target_col)
        
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y)
        mi_importance = dict(zip(feature_cols, mi_scores))
        
        # Calculate random forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = dict(zip(feature_cols, rf.feature_importances_))
            
        # Combine and normalize scores
        combined_scores = {}
        for feature in feature_cols:
            combined_scores[feature] = (
                0.5 * mi_importance[feature] / max(mi_scores) +
                0.5 * rf_importance[feature] / max(rf.feature_importances_)
            )
        
        # Sort and select top features
        sorted_features = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_features]
        
        self.feature_importances = {
            'top_features': dict(sorted_features),
            'total_feature_count': len(feature_cols),
            'average_importance': np.mean(list(combined_scores.values())),
            'method_comparison': {
                'mutual_information': mi_importance,
                'random_forest': rf_importance
            }
        }
        
        return self.feature_importances

    def calculate_city_risk_scores(self, feature_importances=None):
        """
        Calculate city-level risk scores using feature importance and statistics.
        
        Parameters
        ----------
        feature_importances : Dict[str, any], optional
            Pre-calculated feature importances
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing city-level risk scores and related metrics
        """
        if feature_importances is None:
            if self.feature_importances is None:
                feature_importances = self.calculate_feature_importance()
            else:
                feature_importances = self.feature_importances
        
        top_features = list(feature_importances['top_features'].keys())
        
        # Create aggregation dictionary
        agg_dict = {feature: ['mean', 'std'] for feature in top_features}
        agg_dict['arbovirus_bool'] = ['mean', 'sum', 'count']
        
        # Calculate city-level statistics
        city_risk_df = self.df.groupby('city').agg(agg_dict)
        
        # Flatten column names
        city_risk_df.columns = [f"{col[0]}_{col[1]}" for col in city_risk_df.columns]
        city_risk_df = city_risk_df.reset_index()
        
        # Verify required columns exist
        feature_mean_cols = [f"{f}_mean" for f in top_features]
        missing_cols = [col for col in feature_mean_cols if col not in city_risk_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns after aggregation: {missing_cols}")
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(city_risk_df[feature_mean_cols])
        
        # Calculate weights from feature importance
        weights = np.array(list(feature_importances['top_features'].values()))
        weights = weights / np.sum(weights)
        
        # Calculate base risk score
        base_risk_score = np.dot(normalized_features, weights)
        
        # Adjust risk score with arbovirus statistics
        city_risk_df['risk_score'] = (
            base_risk_score * 
            (1 + np.log1p(city_risk_df['arbovirus_bool_sum']) / 
             np.log1p(city_risk_df['arbovirus_bool_count']))
        )
        
        # Scale to 0-100 range
        min_score = city_risk_df['risk_score'].min()
        max_score = city_risk_df['risk_score'].max()
        city_risk_df['risk_score'] = (
            (city_risk_df['risk_score'] - min_score) / 
            (max_score - min_score) * 100
        )
        
        # Add risk categories
        city_risk_df['risk_category'] = pd.qcut(
            city_risk_df['risk_score'], 
            q=3, 
            labels=['Low', 'Medium', 'High']
        )
        
        self.city_risk_scores = city_risk_df
        return city_risk_df
    
    def generate_risk_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive risk assessment report.
        
        Returns
        -------
        Dict[str, any]
            Dictionary containing detailed risk assessment metrics and insights
        """
        if self.city_risk_scores is None:
            self.calculate_city_risk_scores()
            
        if self.feature_importances is None:
            self.calculate_feature_importance()
        
        # Risk distribution analysis
        risk_levels = pd.qcut(
            self.city_risk_scores['risk_score'], 
            q=3, 
            labels=['Low', 'Medium', 'High']
        )
        
        # Calculate temporal trends
        temporal_trends = self.df.groupby(
            pd.Grouper(key='date', freq='M')
        )['arbovirus_bool'].mean()
        
        return {
            'summary_statistics': {
                'total_cities': len(self.city_risk_scores),
                'mean_risk_score': self.city_risk_scores['risk_score'].mean(),
                'median_risk_score': self.city_risk_scores['risk_score'].median(),
                'std_risk_score': self.city_risk_scores['risk_score'].std()
            },
            'risk_distribution': {
                'by_level': risk_levels.value_counts().to_dict(),
                'cities_by_risk_level': {
                    level: self.city_risk_scores[
                        self.city_risk_scores['risk_category'] == level
                    ]['city'].tolist()
                    for level in ['Low', 'Medium', 'High']
                }
            },
            'feature_importance': {
                'top_features': self.feature_importances['top_features'],
                'average_importance': self.feature_importances['average_importance']
            },
            'temporal_analysis': {
                'monthly_trend': temporal_trends.to_dict(),
                'peak_month': temporal_trends.idxmax(),
                'lowest_month': temporal_trends.idxmin()
            }
        }