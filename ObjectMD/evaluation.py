import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd

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

    # ... [other unchanged methods] ...

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

    def evaluate_on_file(self, test_file, video_name=None):
        print(f"\nEvaluating on external file: {test_file}")
        df_test = pd.read_csv(test_file)

        # Filter to specific video
        if video_name is not None and "video_name" in df_test.columns:
            df_test = df_test[df_test["video_name"] == video_name]
            print(f"Filtered to video: {video_name}, {len(df_test)} frames")

        # Create engineered features only if base columns exist
        if 'smoothed_box_speed_3' in df_test.columns and 'box_speed' in df_test.columns:
            df_test['speed_ratio'] = df_test['smoothed_box_speed_3'] / (df_test['box_speed'] + 1e-8)
            df_test['speed_diff'] = df_test['box_speed'] - df_test['smoothed_box_speed_3']

        if 'avg_hand_to_box_dist' in df_test.columns and 'box_speed' in df_test.columns:
            df_test['dist_speed_ratio'] = df_test['avg_hand_to_box_dist'] / (df_test['box_speed'] + 1e-8)

        if 'box_confidence' in df_test.columns and 'box_speed' in df_test.columns:
            df_test['confidence_speed'] = df_test['box_confidence'] * df_test['box_speed']

                # Additional engineered features
        if 'box_speed' in df_test.columns and 'avg_hand_to_box_dist' in df_test.columns:
            df_test['movement_duration'] = df_test['smoothed_box_speed_3'] * df_test['avg_hand_to_box_dist']

    
        # Rolling features
        if 'frame_index' in df_test.columns and 'box_speed' in df_test.columns and len(df_test) > 10:
            df_test = df_test.sort_values('frame_index')
            window_size = min(5, len(df_test) // 10)
            df_test['speed_rolling_mean'] = df_test['box_speed'].rolling(window=window_size, min_periods=1).mean()
            df_test['speed_rolling_std'] = df_test['box_speed'].rolling(window=window_size, min_periods=1).std().fillna(0)

        # Filter valid features
        all_possible_features = [
            'smoothed_box_speed_3', 'box_speed', 'avg_hand_to_box_dist', 'box_confidence',
            'speed_ratio', 'speed_diff', 'dist_speed_ratio', 'confidence_speed',
            'speed_rolling_mean', 'speed_rolling_std'
        ]
        available_features = [f for f in all_possible_features if f in df_test.columns]

        missing = set(self.feature_columns) - set(available_features)
        if missing:
            print(f"Warning: The following features are missing and will be excluded: {missing}")

        if len(available_features) < 2:
            print("Temporal Evaluation:\nInsufficient data to compute temporal evaluation.")
            return

        df_test = df_test.dropna(subset=available_features)
        if df_test.empty:
            print("Temporal Evaluation:\nNo valid rows after dropping missing features.")
            return

        X_eval = self.scaler.transform(df_test[available_features])
        y_eval = df_test[self.target_column] if self.target_column in df_test.columns else None

        y_pred = self.best_model.predict(X_eval)

        if y_eval is not None:
            self.temporal_evaluation(y_eval.tolist(), y_pred.tolist())
        else:
            print("Ground-truth 'is_moving' not available, skipping temporal evaluation.")

    def temporal_evaluation(self, y_true, y_pred):
        """Evaluate temporal alignment between predicted and true movement segments"""
        def find_segment(y):
            start, end = None, None
            for i, val in enumerate(y):
                if val == 1 and start is None:
                    start = i
                if val == 1:
                    end = i
            return start, end

        start_true, end_true = find_segment(y_true)
        
        start_pred, end_pred = find_segment(y_pred)

        print("\nTemporal Evaluation:")
        if start_true is None or end_true is None:
            print("No ground truth movement detected.")
            return
        if start_pred is None or end_pred is None:
            print("No predicted movement detected.")
            return

        print(f"True Start Frame: {start_true} | Predicted Start Frame: {start_pred}")
        print(f"Start Frame Error: {abs(start_true - start_pred)} frames")

        print(f"True End Frame: {end_true} | Predicted End Frame: {end_pred}")
        print(f"End Frame Error: {abs(end_true - end_pred)} frames")

        # Compute temporal Intersection over Union
        inter_start = max(start_true, start_pred)
        inter_end = min(end_true, end_pred)
        intersection = max(0, inter_end - inter_start + 1)
        union = max(end_true, end_pred) - min(start_true, start_pred) + 1
        iou = intersection / union if union > 0 else 0.0
        print(f"Temporal IoU: {iou:.3f}")
        
              # visualization
        plt.figure(figsize=(10, 2))
        plt.title("Temporal Boundary Comparison")
        plt.axvline(start_true, color='green', linestyle='--', label='True Start')
        plt.axvline(end_true, color='green', linestyle='-', label='True End')
        plt.axvline(start_pred, color='red', linestyle='--', label='Predicted Start')
        plt.axvline(end_pred, color='red', linestyle='-', label='Predicted End')

        plt.xlabel("Frame Index")
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=4)
        plt.text(0.95, 0.9, f"IoU: {iou:.3f}", ha='right', va='center',
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray'))

        plt.tight_layout()
        plt.savefig("reports/figures/temporal_boundary_summary.png")
        plt.close()




    # Usage example:
if __name__ == "__main__":
    # Load pre-trained model and scaler
    classifier = BoxMovementClassifier(data_path='data/features/features.csv')
    classifier.best_model = joblib.load('models/best_box_movement_classifier.pkl')
    classifier.scaler = joblib.load('models/feature_scaler.pkl')
    
    # Set feature columns to match training
    classifier.feature_columns = [
        'smoothed_box_speed_3', 'box_speed', 'avg_hand_to_box_dist', 'box_confidence',
        'speed_ratio', 'speed_diff', 'dist_speed_ratio', 'confidence_speed'
    ]
    
    # Run evaluation on test file
    classifier.evaluate_on_file(
        test_file='data/features/features.csv',
        video_name='gd_0022'
    )

    