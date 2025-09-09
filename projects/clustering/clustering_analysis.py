#!/usr/bin/env python3
"""
Customer Segmentation Clustering Project
========================================

This script demonstrates a complete unsupervised learning pipeline for 
customer segmentation using various clustering algorithms.

Author: ML Portfolio
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """
    A complete customer segmentation system using clustering algorithms.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.clustering_results = {}
        self.optimal_k = None
        
    def load_data(self, file_path):
        """Load and return the dataset."""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print("Creating sample customer data for demonstration...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample customer segmentation data for demonstration."""
        np.random.seed(42)
        n_samples = 2000
        
        # Create customer segments with different characteristics
        segments = [
            {'size': 300, 'recency': (1, 30), 'frequency': (8, 15), 'monetary': (800, 2000), 'name': 'Champions'},
            {'size': 460, 'recency': (30, 90), 'frequency': (4, 8), 'monetary': (400, 800), 'name': 'Loyal'},
            {'size': 360, 'recency': (15, 60), 'frequency': (2, 6), 'monetary': (200, 600), 'name': 'Potential'},
            {'size': 400, 'recency': (90, 180), 'frequency': (2, 5), 'monetary': (300, 700), 'name': 'At Risk'},
            {'size': 480, 'recency': (60, 365), 'frequency': (1, 3), 'monetary': (50, 300), 'name': 'Price Sensitive'}
        ]
        
        data_list = []
        customer_id = 1
        
        for segment in segments:
            for _ in range(segment['size']):
                # Generate customer data based on segment characteristics
                recency = np.random.randint(segment['recency'][0], segment['recency'][1])
                frequency = np.random.randint(segment['frequency'][0], segment['frequency'][1])
                monetary = np.random.randint(segment['monetary'][0], segment['monetary'][1])
                
                # Add some noise and derived features
                avg_order_value = monetary / frequency if frequency > 0 else 0
                avg_order_value += np.random.normal(0, 20)  # Add noise
                
                product_diversity = np.random.randint(1, min(8, frequency + 2))
                seasonal_purchases = np.random.randint(1, 5)
                days_since_first_purchase = recency + np.random.randint(30, 730)
                
                # Customer age and gender for additional demographics
                age = np.random.randint(18, 70)
                gender = np.random.choice(['M', 'F'])
                
                data_list.append({
                    'CustomerID': customer_id,
                    'Recency': recency,
                    'Frequency': frequency,
                    'Monetary': monetary,
                    'AvgOrderValue': max(avg_order_value, 10),  # Minimum order value
                    'ProductDiversity': product_diversity,
                    'SeasonalPurchases': seasonal_purchases,
                    'DaysSinceFirstPurchase': days_since_first_purchase,
                    'Age': age,
                    'Gender': gender,
                    'TrueSegment': segment['name']  # For validation (normally not available)
                })
                customer_id += 1
        
        self.data = pd.DataFrame(data_list)
        print(f"Sample data created: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n=== DATA EXPLORATION ===")
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nDataset Description:")
        print(self.data.describe())
        
        # Create comprehensive visualizations
        plt.figure(figsize=(20, 15))
        
        # RFM distribution plots
        rfm_features = ['Recency', 'Frequency', 'Monetary']
        for i, feature in enumerate(rfm_features, 1):
            plt.subplot(3, 4, i)
            plt.hist(self.data[feature], bins=30, alpha=0.7, color='skyblue')
            plt.title(f'{feature} Distribution')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        
        # Box plots for key features
        plt.subplot(3, 4, 4)
        plt.boxplot([self.data['Recency'], self.data['Frequency'], self.data['Monetary']], 
                   labels=['Recency', 'Frequency', 'Monetary'])
        plt.title('RFM Features Box Plot')
        plt.yscale('log')
        
        # Correlation heatmap
        plt.subplot(3, 4, 5)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['CustomerID']]
        correlation_matrix = self.data[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlations')
        
        # Scatter plots for feature relationships
        plt.subplot(3, 4, 6)
        plt.scatter(self.data['Frequency'], self.data['Monetary'], alpha=0.6)
        plt.xlabel('Frequency')
        plt.ylabel('Monetary')
        plt.title('Frequency vs Monetary')
        
        plt.subplot(3, 4, 7)
        plt.scatter(self.data['Recency'], self.data['Monetary'], alpha=0.6)
        plt.xlabel('Recency')
        plt.ylabel('Monetary')
        plt.title('Recency vs Monetary')
        
        plt.subplot(3, 4, 8)
        plt.scatter(self.data['Recency'], self.data['Frequency'], alpha=0.6)
        plt.xlabel('Recency')
        plt.ylabel('Frequency')
        plt.title('Recency vs Frequency')
        
        # Additional feature distributions
        plt.subplot(3, 4, 9)
        plt.hist(self.data['AvgOrderValue'], bins=30, alpha=0.7, color='lightgreen')
        plt.title('Average Order Value Distribution')
        plt.xlabel('Average Order Value')
        
        plt.subplot(3, 4, 10)
        plt.hist(self.data['ProductDiversity'], bins=range(1, 9), alpha=0.7, color='orange')
        plt.title('Product Diversity Distribution')
        plt.xlabel('Product Categories')
        
        # Age and gender analysis
        plt.subplot(3, 4, 11)
        age_bins = [18, 25, 35, 45, 55, 70]
        self.data['AgeGroup'] = pd.cut(self.data['Age'], bins=age_bins, include_lowest=True)
        age_monetary = self.data.groupby('AgeGroup')['Monetary'].mean()
        age_monetary.plot(kind='bar')
        plt.title('Average Spending by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Average Monetary Value')
        plt.xticks(rotation=45)
        
        plt.subplot(3, 4, 12)
        gender_stats = self.data.groupby('Gender')[['Frequency', 'Monetary']].mean()
        gender_stats.plot(kind='bar')
        plt.title('Frequency and Monetary by Gender')
        plt.xlabel('Gender')
        plt.xticks(rotation=0)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('customer_data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """Preprocess data for clustering."""
        print("\n=== DATA PREPROCESSING ===")
        
        # Select features for clustering
        clustering_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 
                             'ProductDiversity', 'DaysSinceFirstPurchase', 'Age']
        
        self.X = self.data[clustering_features].copy()
        
        # Handle any missing values
        self.X = self.X.fillna(self.X.mean())
        
        # Remove outliers using IQR method
        Q1 = self.X.quantile(0.25)
        Q3 = self.X.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        outlier_mask = ((self.X < lower_bound) | (self.X > upper_bound)).any(axis=1)
        self.X_clean = self.X[~outlier_mask]
        self.data_clean = self.data[~outlier_mask]
        
        print(f"Removed {outlier_mask.sum()} outliers ({outlier_mask.sum()/len(self.X)*100:.1f}%)")
        print(f"Clean dataset shape: {self.X_clean.shape}")
        
        # Scale the features
        self.X_scaled = self.scaler.fit_transform(self.X_clean)
        
        # Apply PCA for visualization
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        print(f"Features used: {clustering_features}")
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        print(f"Total variance explained: {self.pca.explained_variance_ratio_.sum():.3f}")
    
    def find_optimal_clusters(self):
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        print("\n=== FINDING OPTIMAL CLUSTERS ===")
        
        k_range = range(2, 11)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, cluster_labels))
        
        # Plot elbow curve and silhouette scores
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(k_range, inertias, 'bo-')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(k_range, silhouette_scores, 'ro-')
        plt.title('Silhouette Score')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)
        
        # Find optimal k (highest silhouette score)
        self.optimal_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"Optimal number of clusters: {self.optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.3f}")
        
        # Detailed silhouette analysis for optimal k
        plt.subplot(1, 3, 3)
        kmeans_optimal = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans_optimal.fit_predict(self.X_scaled)
        
        silhouette_avg = silhouette_score(self.X_scaled, cluster_labels)
        plt.title(f'Silhouette Analysis (k={self.optimal_k})')
        plt.xlabel('Silhouette Coefficient Values')
        plt.ylabel('Cluster Label')
        
        y_lower = 10
        for i in range(self.optimal_k):
            cluster_silhouette_vals = silhouette_scores[cluster_labels == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.nipy_spectral(float(i) / self.optimal_k)
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            y_lower = y_upper + 10
        
        plt.axvline(x=silhouette_avg, color="red", linestyle="--", 
                   label=f'Average Score: {silhouette_avg:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.optimal_k
    
    def apply_clustering_algorithms(self):
        """Apply different clustering algorithms."""
        print("\n=== APPLYING CLUSTERING ALGORITHMS ===")
        
        algorithms = {
            'K-Means': KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10),
            'Hierarchical': AgglomerativeClustering(n_clusters=self.optimal_k, linkage='ward'),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }
        
        for name, algorithm in algorithms.items():
            print(f"\nApplying {name}...")
            
            if name == 'DBSCAN':
                # DBSCAN doesn't use n_clusters, so we need to tune eps
                labels = algorithm.fit_predict(self.X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                print(f"Estimated number of clusters: {n_clusters}")
                print(f"Estimated number of noise points: {n_noise}")
            else:
                labels = algorithm.fit_predict(self.X_scaled)
                n_clusters = self.optimal_k
            
            # Calculate metrics
            if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                silhouette_avg = silhouette_score(self.X_scaled, labels)
                davies_bouldin = davies_bouldin_score(self.X_scaled, labels)
            else:
                silhouette_avg = -1
                davies_bouldin = float('inf')
            
            self.clustering_results[name] = {
                'algorithm': algorithm,
                'labels': labels,
                'n_clusters': n_clusters,
                'silhouette_score': silhouette_avg,
                'davies_bouldin_score': davies_bouldin
            }
            
            print(f"Silhouette Score: {silhouette_avg:.3f}")
            print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    def visualize_clusters(self):
        """Visualize clustering results."""
        print("\n=== VISUALIZING CLUSTERS ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        # Original data distribution (PCA)
        axes[0].scatter(self.X_pca[:, 0], self.X_pca[:, 1], alpha=0.6)
        axes[0].set_title('Original Data (PCA)')
        axes[0].set_xlabel('First Principal Component')
        axes[0].set_ylabel('Second Principal Component')
        
        # Clustering results
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, (name, result) in enumerate(self.clustering_results.items(), 1):
            labels = result['labels']
            
            # Handle noise points in DBSCAN
            unique_labels = set(labels)
            if -1 in unique_labels:
                # Plot noise points in black
                noise_mask = labels == -1
                axes[i].scatter(self.X_pca[noise_mask, 0], self.X_pca[noise_mask, 1], 
                              c='black', marker='x', s=50, alpha=0.6, label='Noise')
                unique_labels.remove(-1)
            
            # Plot clusters
            for j, label in enumerate(sorted(unique_labels)):
                cluster_mask = labels == label
                axes[i].scatter(self.X_pca[cluster_mask, 0], self.X_pca[cluster_mask, 1], 
                              c=colors[j % len(colors)], alpha=0.6, 
                              label=f'Cluster {label}' if label != -1 else 'Noise')
            
            axes[i].set_title(f'{name} Clustering\n'
                            f'Silhouette: {result["silhouette_score"]:.3f}')
            axes[i].set_xlabel('First Principal Component')
            axes[i].set_ylabel('Second Principal Component')
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Feature comparison for K-Means (best performing)
        kmeans_labels = self.clustering_results['K-Means']['labels']
        
        # RFM comparison
        axes[4].clear()
        rfm_by_cluster = self.data_clean.groupby(kmeans_labels)[['Recency', 'Frequency', 'Monetary']].mean()
        rfm_by_cluster.plot(kind='bar', ax=axes[4])
        axes[4].set_title('RFM Analysis by Cluster (K-Means)')
        axes[4].set_xlabel('Cluster')
        axes[4].set_ylabel('Average Value')
        axes[4].legend()
        axes[4].tick_params(axis='x', rotation=0)
        
        # Cluster size distribution
        axes[5].clear()
        cluster_sizes = pd.Series(kmeans_labels).value_counts().sort_index()
        axes[5].pie(cluster_sizes.values, labels=[f'Cluster {i}' for i in cluster_sizes.index], 
                   autopct='%1.1f%%')
        axes[5].set_title('Cluster Size Distribution (K-Means)')
        
        plt.tight_layout()
        plt.savefig('clustering_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_segments(self):
        """Analyze and interpret customer segments."""
        print("\n=== SEGMENT ANALYSIS ===")
        
        # Use K-Means results for detailed analysis
        kmeans_labels = self.clustering_results['K-Means']['labels']
        
        # Add cluster labels to clean data
        analysis_data = self.data_clean.copy()
        analysis_data['Cluster'] = kmeans_labels
        
        # Calculate segment characteristics
        segment_summary = analysis_data.groupby('Cluster').agg({
            'Recency': ['mean', 'std'],
            'Frequency': ['mean', 'std'],
            'Monetary': ['mean', 'std'],
            'AvgOrderValue': ['mean', 'std'],
            'ProductDiversity': ['mean', 'std'],
            'Age': ['mean', 'std'],
            'CustomerID': 'count'
        }).round(2)
        
        segment_summary.columns = ['_'.join(col).strip() for col in segment_summary.columns]
        print("\nSegment Summary Statistics:")
        print(segment_summary)
        
        # Detailed segment profiling
        print("\n=== DETAILED SEGMENT PROFILES ===")
        
        for cluster in sorted(analysis_data['Cluster'].unique()):
            cluster_data = analysis_data[analysis_data['Cluster'] == cluster]
            size = len(cluster_data)
            percentage = size / len(analysis_data) * 100
            
            print(f"\n--- CLUSTER {cluster} ---")
            print(f"Size: {size} customers ({percentage:.1f}%)")
            print(f"Recency: {cluster_data['Recency'].mean():.1f} ± {cluster_data['Recency'].std():.1f} days")
            print(f"Frequency: {cluster_data['Frequency'].mean():.1f} ± {cluster_data['Frequency'].std():.1f} purchases")
            print(f"Monetary: ${cluster_data['Monetary'].mean():.0f} ± ${cluster_data['Monetary'].std():.0f}")
            print(f"Avg Order Value: ${cluster_data['AvgOrderValue'].mean():.0f}")
            print(f"Product Diversity: {cluster_data['ProductDiversity'].mean():.1f} categories")
            print(f"Average Age: {cluster_data['Age'].mean():.1f} years")
            
            # Gender distribution
            gender_dist = cluster_data['Gender'].value_counts(normalize=True) * 100
            print(f"Gender: {gender_dist.to_dict()}")
        
        # Business recommendations
        self.generate_business_recommendations(analysis_data)
        
        return analysis_data
    
    def generate_business_recommendations(self, analysis_data):
        """Generate business recommendations for each segment."""
        print("\n=== BUSINESS RECOMMENDATIONS ===")
        
        # Define segment characteristics and recommendations
        segment_recommendations = {
            0: {
                'name': 'Champions',
                'description': 'Best customers with high value and frequency',
                'strategy': 'Reward loyalty, VIP treatment, referral programs'
            },
            1: {
                'name': 'Loyal Customers',
                'description': 'Regular customers with good value',
                'strategy': 'Upselling, cross-selling, loyalty programs'
            },
            2: {
                'name': 'Potential Loyalists',
                'description': 'Recent customers with potential',
                'strategy': 'Engagement campaigns, personalized offers'
            },
            3: {
                'name': 'At Risk',
                'description': 'Previously valuable, now declining',
                'strategy': 'Win-back campaigns, special offers, surveys'
            },
            4: {
                'name': 'Price Sensitive',
                'description': 'Low spenders, price-conscious',
                'strategy': 'Discount offers, value propositions'
            }
        }
        
        # Analyze each cluster and assign business meaning
        for cluster in sorted(analysis_data['Cluster'].unique()):
            cluster_data = analysis_data[analysis_data['Cluster'] == cluster]
            
            avg_recency = cluster_data['Recency'].mean()
            avg_frequency = cluster_data['Frequency'].mean()
            avg_monetary = cluster_data['Monetary'].mean()
            
            print(f"\nCluster {cluster} Business Profile:")
            
            # Assign segment based on RFM characteristics
            if avg_frequency >= 6 and avg_monetary >= 600 and avg_recency <= 60:
                segment_type = 'Champions'
            elif avg_frequency >= 4 and avg_monetary >= 400:
                segment_type = 'Loyal Customers'
            elif avg_recency <= 90 and avg_monetary >= 200:
                segment_type = 'Potential Loyalists'
            elif avg_recency > 90 and avg_monetary >= 300:
                segment_type = 'At Risk'
            else:
                segment_type = 'Price Sensitive'
            
            print(f"Segment Type: {segment_type}")
            print(f"Key Characteristics:")
            print(f"  - Average last purchase: {avg_recency:.0f} days ago")
            print(f"  - Purchase frequency: {avg_frequency:.1f} times")
            print(f"  - Average spending: ${avg_monetary:.0f}")
            
            # Find matching recommendation
            for rec_cluster, rec_info in segment_recommendations.items():
                if rec_info['name'] == segment_type:
                    print(f"Recommended Strategy: {rec_info['strategy']}")
                    break
    
    def model_validation(self):
        """Validate clustering model quality."""
        print("\n=== MODEL VALIDATION ===")
        
        # Compare different algorithms
        comparison_data = []
        for name, result in self.clustering_results.items():
            comparison_data.append({
                'Algorithm': name,
                'Clusters': result['n_clusters'],
                'Silhouette Score': result['silhouette_score'],
                'Davies-Bouldin Score': result['davies_bouldin_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Silhouette Score', ascending=False)
        
        print("\nAlgorithm Comparison:")
        print(comparison_df.round(3))
        
        # Best model
        best_algorithm = comparison_df.iloc[0]['Algorithm']
        print(f"\nBest performing algorithm: {best_algorithm}")
        
        # Stability test (run multiple times with different random states)
        print("\nStability Test (K-Means with different random states):")
        silhouette_scores = []
        
        for random_state in range(5):
            kmeans = KMeans(n_clusters=self.optimal_k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            score = silhouette_score(self.X_scaled, labels)
            silhouette_scores.append(score)
        
        print(f"Silhouette scores: {[f'{score:.3f}' for score in silhouette_scores]}")
        print(f"Mean: {np.mean(silhouette_scores):.3f}")
        print(f"Std: {np.std(silhouette_scores):.3f}")
        
        if np.std(silhouette_scores) < 0.05:
            print("✓ Model shows good stability across different initializations")
        else:
            print("⚠ Model shows some instability - consider different parameters")

def main():
    """Main execution function."""
    print("=== CUSTOMER SEGMENTATION CLUSTERING PROJECT ===")
    
    # Initialize segmentation system
    segmentation = CustomerSegmentation()
    
    # Load data (will create sample data if file not found)
    segmentation.load_data('customer_data.csv')
    
    # Perform analysis
    segmentation.explore_data()
    segmentation.preprocess_data()
    optimal_k = segmentation.find_optimal_clusters()
    segmentation.apply_clustering_algorithms()
    segmentation.visualize_clusters()
    analysis_data = segmentation.analyze_segments()
    segmentation.model_validation()
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Optimal number of clusters: {optimal_k}")
    print("Check generated plots:")
    print("- customer_data_exploration.png")
    print("- optimal_clusters_analysis.png") 
    print("- clustering_visualization.png")
    
    return segmentation

if __name__ == "__main__":
    main()