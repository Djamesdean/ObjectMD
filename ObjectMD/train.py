import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
import xgboost as xgb

warnings.filterwarnings('ignore')

class BoxMovementClassifier:
    def __init__(self, data_path=None, df=None):
     
        if data_path:
            self.df = pd.read_csv(data_path)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError(" data_path or df must be provided")
        
        self.feature_columns = [
            'smoothed_box_speed_3', 'box_speed', 'avg_hand_to_box_dist', 'box_confidence'
        ]
        self.target_column = 'is_moving'
        self.scaler = None
        self.best_model = None
        self.best_score = 0
        self.results = {}
        
    def prepare_data(self):
        
        # Check if target column exists
        if self.target_column not in self.df.columns:
            print(f"Target column '{self.target_column}' not found. Creating based on box_speed threshold...")
            # Create target based on box speed (you can adjust this threshold)
            speed_threshold = self.df['box_speed'].median()  # or use a fixed value like 0.1
            self.df[self.target_column] = (self.df['box_speed'] > speed_threshold).astype(int)
        
        self.df = self.df.dropna(subset=self.feature_columns + [self.target_column])
        
        # Remove outliers using IQR method
        for col in self.feature_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self.df
    
    def feature_engineering(self):
        
        #Creating more features based on existing ones
        
        # Speed ratios and differences
        self.df['speed_ratio'] = self.df['smoothed_box_speed_3'] / (self.df['box_speed'] + 1e-8)
        self.df['speed_diff'] = self.df['box_speed'] - self.df['smoothed_box_speed_3']
        
        # Distance-speed interactions
        self.df['dist_speed_ratio'] = self.df['avg_hand_to_box_dist'] / (self.df['box_speed'] + 1e-8)
        self.df['confidence_speed'] = self.df['box_confidence'] * self.df['box_speed']
        
        # Rolling statistics (if we have sequential data)
        if len(self.df) > 10:
            self.df = self.df.sort_values('frame_index') if 'frame_index' in self.df.columns else self.df
            window_size = min(5, len(self.df) // 10)
            self.df['speed_rolling_mean'] = self.df['box_speed'].rolling(window=window_size, min_periods=1).mean()
            self.df['speed_rolling_std'] = self.df['box_speed'].rolling(window=window_size, min_periods=1).std().fillna(0)
        
        # Update feature columns
        self.feature_columns.extend([
            'speed_ratio', 'speed_diff', 'dist_speed_ratio', 'confidence_speed'
        ])
        
        if 'speed_rolling_mean' in self.df.columns:
            self.feature_columns.extend(['speed_rolling_mean', 'speed_rolling_std'])
        
        print(f"Total features: {len(self.feature_columns)}")
        
    def get_classifiers(self):
        """Define all classifiers to test"""
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'Naive Bayes': GaussianNB(),
        }
        return classifiers
    
    def evaluate_classifiers(self, test_size=0.2, cv_folds=5):
        """Evaluate all classifiers"""
        print("Evaluating classifiers...")
        
        # Prepare features and target
        X = self.df[self.feature_columns]
        y = self.df[self.target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale the features
        self.scaler = RobustScaler()  # RobustScaler is less sensitive to outliers
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get classifiers
        classifiers = self.get_classifiers()
        
        # Evaluate each classifier
        results = []
        for name, clf in classifiers.items():
            print(f"Testing {name}...")
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv_folds, scoring='f1')
                
                # Fit and predict
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results.append({
                    'Classifier': name,
                    'CV_F1_Mean': cv_scores.mean(),
                    'CV_F1_Std': cv_scores.std(),
                    'Test_Accuracy': accuracy,
                    'Test_Precision': precision,
                    'Test_Recall': recall,
                    'Test_F1': f1
                })
                
                # Store the best model
                if cv_scores.mean() > self.best_score:
                    self.best_score = cv_scores.mean()
                    self.best_model = clf
                    self.best_model_name = name
                
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
                continue
        
        # Create results DataFrame
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values('CV_F1_Mean', ascending=False)
        
        return self.results_df, X_test_scaled, y_test
    
    def hyperparameter_tuning(self, top_n=3):
        """Perform hyperparameter tuning on top N classifiers"""
        print(f"Performing hyperparameter tuning on top {top_n} classifiers...")
        
        X = self.df[self.feature_columns]
        y = self.df[self.target_column]
        X_scaled = self.scaler.transform(X)
        
        # Get top classifiers
        top_classifiers = self.results_df.head(top_n)['Classifier'].tolist()
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'SVM (RBF)': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        best_tuned_score = 0
        best_tuned_model = None
        
        for clf_name in top_classifiers:
            if clf_name in param_grids:
                print(f"Tuning {clf_name}...")
                
                # Get base classifier
                base_clf = self.get_classifiers()[clf_name]
                
                # Grid search
                grid_search = GridSearchCV(
                    base_clf, param_grids[clf_name], 
                    cv=5, scoring='f1', n_jobs=-1
                )
                
                grid_search.fit(X_scaled, y)
                
                if grid_search.best_score_ > best_tuned_score:
                    best_tuned_score = grid_search.best_score_
                    best_tuned_model = grid_search.best_estimator_
                    self.best_model = best_tuned_model
                    self.best_model_name = f"{clf_name} (Tuned)"
                
                print(f"{clf_name} best score: {grid_search.best_score_:.4f}")
                print(f"Best parameters: {grid_search.best_params_}")
    

    
    def plot_results(self):
        """Plot classifier comparison"""
        plt.figure(figsize=(12, 8))
        
        # Plot CV F1 scores
        plt.subplot(2, 2, 1)
        sns.barplot(data=self.results_df.head(10), x='CV_F1_Mean', y='Classifier')
        plt.title('Cross-Validation F1 Scores')
        
        # Plot test accuracy
        plt.subplot(2, 2, 2)
        sns.barplot(data=self.results_df.head(10), x='Test_Accuracy', y='Classifier')
        plt.title('Test Accuracy')
        
        # Plot test F1
        plt.subplot(2, 2, 3)
        sns.barplot(data=self.results_df.head(10), x='Test_F1', y='Classifier')
        plt.title('Test F1 Score')
        
        # Plot precision vs recall
        plt.subplot(2, 2, 4)
        plt.scatter(self.results_df['Test_Precision'], self.results_df['Test_Recall'])
        for i, txt in enumerate(self.results_df['Classifier']):
            plt.annotate(txt[:10], (self.results_df['Test_Precision'].iloc[i], 
                                  self.results_df['Test_Recall'].iloc[i]), 
                        fontsize=8, rotation=45)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision vs Recall')
        
        plt.tight_layout()
        plt.savefig("reports/figures/classifier_comparison.png")
        plt.close()
    
    def final_evaluation(self, X_test, y_test):
        """Final evaluation of the best model"""
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"{'='*50}")
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        
        # Detailed metrics
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Moving', 'Moving'], 
                   yticklabels=['Not Moving', 'Moving'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("reports/figures/confusion_matrix.png")
        plt.close()
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Box Movement Classification Analysis")
        print("="*60)
        
        # 1. Prepare data
        self.prepare_data()
        
        # 2. Feature engineering
        self.feature_engineering()
        
        # 3. Evaluate classifiers
        results_df, X_test, y_test = self.evaluate_classifiers()
        
        # 4. Display results
        print("\nClassifier Comparison Results:")
        print(results_df.round(4))
        
        # 5. Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # 6. Plot results
        self.plot_results()
        
        # 7. Final evaluation
        self.final_evaluation(X_test, y_test)
        
        return self.best_model, results_df

# Usage example:
if __name__ == "__main__":
  
    classifier = BoxMovementClassifier(data_path='data/features/features.csv')
    
   
    
    # Run complete analysis
    best_model, results = classifier.run_complete_analysis()
    
    # Save the best model
    
    joblib.dump(classifier.best_model, 'models/best_box_movement_classifier.pkl')
    joblib.dump(classifier.scaler, 'models/feature_scaler.pkl')
    
    print("Analysis complete! The best model has been identified and can be saved for future use.")