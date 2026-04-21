import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    #Compute Semantic Stability & Accuracy Sensitivity Metrics

    import json
    import pandas as pd
    import numpy as np
    from typing import Dict, List, Tuple
    from collections import defaultdict

    return np, pd


@app.cell
def _(pd):
    #Load results saved from prompts.py

    results_df = pd.read_csv("data/raw_results/inference_results.csv")

    print(" Loaded inference results:")
    print(f"   Rows: {len(results_df)}")
    print(f"   Models: {results_df['model'].unique().tolist()}")
    print(f"   Templates: {len(results_df['template_name'].unique())}")
    return (results_df,)


@app.cell
def _(np, pd, results_df):
    #Metric 1 - Semantic Stability (StabSem)

    print("COMPUTING: SEMANTIC STABILITY (StabSem)")

    def compute_semantic_stability_embedding() -> pd.DataFrame:
        """
        Compute semantic stability using sentence embeddings (Week 2, Approach 1)
    
        StabSem = average pairwise semantic similarity between responses to 
                  the same question under different prompt templates
    
        Range: [0, 1] where:
        - 1.0 = always semantically identical responses
        - 0.5 = moderate semantic drift
        - 0.0 = completely different meanings
        """
    
        # If you have sentence-transformers installed:
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')
    
        # For now: PLACEHOLDER implementation
        # Replace with real embedding computation
    
        stability_scores = []
    
        for item_id in results_df['item_id'].unique():
            item_results = results_df[results_df['item_id'] == item_id]
        
            for model in item_results['model'].unique():
                model_results = item_results[item_results['model'] == model]
            
                if len(model_results) > 1:
                    responses = model_results['response'].tolist()
                
                    # MOCK: In real implementation, compute embeddings
                    # For now: simple string similarity as proxy
                    similarity_scores = []
                    for i in range(len(responses)):
                        for j in range(i+1, len(responses)):
                            # String similarity (0-1)
                            sim = len(set(responses[i]) & set(responses[j])) / \
                                  max(len(set(responses[i]) | set(responses[j])), 1)
                            similarity_scores.append(sim)
                
                    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.5
                
                    stability_scores.append({
                        'item_id': item_id,
                        'model': model,
                        'StabSem': avg_similarity,
                        'num_templates': len(model_results)
                    })
    
        return pd.DataFrame(stability_scores)

    StabSem_df = compute_semantic_stability_embedding()
    print(f"\n Computed StabSem for {len(StabSem_df)} (model, item) pairs")
    print("\nSample StabSem scores:")
    print(StabSem_df.head(10))

    return (StabSem_df,)


@app.cell
def _(np, pd, results_df):
    #Metric 2 - Accuracy Sensitivity (SensAcc)

    print("\n" + "="*70)
    print("COMPUTING: ACCURACY SENSITIVITY (SensAcc)")
    print("="*70)

    def compute_accuracy_sensitivity() -> pd.DataFrame:
        """
        Compute accuracy sensitivity: how much does model correctness vary
        across different prompt templates?
    
        SensAcc = standard deviation of correctness scores across templates
    
        Range: [0, 1] where:
        - 0.0 = consistent performance (always correct or always wrong)
        - 1.0 = high variability (sometimes correct, sometimes wrong)
        """
    
        sensitivity_scores = []
    
        for item_id in results_df['item_id'].unique():
            item_results = results_df[results_df['item_id'] == item_id]
        
            for model in item_results['model'].unique():
                model_results = item_results[item_results['model'] == model]
            
                correctness_scores = model_results['is_correct'].astype(int).tolist()
            
                # Standard deviation of correctness
                sens = np.std(correctness_scores) if len(correctness_scores) > 1 else 0.0
            
                sensitivity_scores.append({
                    'item_id': item_id,
                    'model': model,
                    'SensAcc': sens,
                    'mean_accuracy': np.mean(correctness_scores),
                    'num_correct': sum(correctness_scores),
                    'num_templates': len(correctness_scores)
                })
    
        return pd.DataFrame(sensitivity_scores)

    SensAcc_df = compute_accuracy_sensitivity()
    print(f"\n Computed SensAcc for {len(SensAcc_df)} (model, item) pairs")
    print("\nSample SensAcc scores:")
    print(SensAcc_df.head(10))

    return (SensAcc_df,)


@app.cell
def _(SensAcc_df, StabSem_df, pd):
    #Metric 3 - Artifact Score (Combined)

    print("\n" + "="*70)
    print("COMPUTING: ARTIFACT SCORE (Combined)")
    print("="*70)

    def compute_artifact_score(sens_df: pd.DataFrame, stab_df: pd.DataFrame) -> pd.DataFrame:
        """
        Artifact Score = SensAcc / (SensAcc + StabSem)
    
        Interpretation:
        - Score near 1.0: High accuracy sensitivity BUT low semantic stability
                          → Likely a MEASUREMENT ARTIFACT
        - Score near 0.0: High accuracy sensitivity BUT high semantic stability
                          → Likely a REAL ROBUSTNESS ISSUE
        - Score near 0.5: Balanced (neutral case)
    
        Example:
        - Item has 30% accuracy variation but 85% semantic stability
        → ArtifactScore = 0.30 / (0.30 + 0.85) = 0.26 (LOW)
        → Interpretation: This is probably real sensitivity, not measurement artifact
    
        - Item has 35% accuracy variation but 92% semantic stability
        → ArtifactScore = 0.35 / (0.35 + 0.92) = 0.28 (LOW)
        → Interpretation: Responses are semantically the same, but marked wrong due to rigid evaluation
        """
    
        artifact_scores = []
    
        # Merge the two dataframes
        merged = sens_df.merge(stab_df, on=['item_id', 'model'], how='inner')
    
        for _, row in merged.iterrows():
            sens = row['SensAcc']
            stab = row['StabSem']
        
            # Avoid division by zero
            artifact_score = sens / (sens + stab) if (sens + stab) > 0 else 0.5
        
            # Categorize
            if artifact_score > 0.7:
                category = "LIKELY_ARTIFACT"
            elif artifact_score > 0.4:
                category = "MIXED"
            else:
                category = "LIKELY_REAL_ISSUE"
        
            artifact_scores.append({
                'item_id': row['item_id'],
                'model': row['model'],
                'SensAcc': sens,
                'StabSem': stab,
                'ArtifactScore': artifact_score,
                'Category': category,
                'mean_accuracy': row['mean_accuracy']
            })
    
        return pd.DataFrame(artifact_scores)

    Article_df = compute_artifact_score(SensAcc_df, StabSem_df)
    print(f"\n Computed ArtifactScore for {len(Article_df)} (model, item) pairs")
    print("\nSample Artifact Scores:")
    print(Article_df.head(15))

    return (Article_df,)


@app.cell
def _(Article_df):
    #Summary Statistics

    print("SUMMARY STATISTICS")

    print("\n1️.  ARTIFACT SCORE DISTRIBUTION:")
    print(Article_df['Category'].value_counts())
    print(f"\n   % predicted as ARTIFACTS: {100*len(Article_df[Article_df['Category']=='LIKELY_ARTIFACT'])/len(Article_df):.1f}%")

    print("\n2.  BY MODEL:")
    for model in Article_df['model'].unique():
        model_data = Article_df[Article_df['model'] == model]
        print(f"\n   {model}:")
        print(f"      Avg ArtifactScore: {model_data['ArtifactScore'].mean():.3f}")
        print(f"      Avg SensAcc:       {model_data['SensAcc'].mean():.3f}")
        print(f"      Avg StabSem:       {model_data['StabSem'].mean():.3f}")
        print(f"      % Artifacts:       {100*len(model_data[model_data['Category']=='LIKELY_ARTIFACT'])/len(model_data):.1f}%")

    print("\n3️.  CORRELATION ANALYSIS:")
    correlation = Article_df['SensAcc'].corr(Article_df['StabSem'])
    print(f"    Correlation (SensAcc vs StabSem): {correlation:.3f}")
    if abs(correlation) < 0.3:
        print("    → WEAK correlation: Metrics are independent (good!)")
    elif abs(correlation) < 0.7:
        print("    → MODERATE correlation: Some relationship")
    else:
        print("    → STRONG correlation: Strong relationship")

    return (model,)


@app.cell
def _(Article_df):
    #Save results

    Article_df.to_csv("data/raw_results/metrics_summary.csv", index=False)
    print("\n Metrics saved to data/raw_results/metrics_summary.csv")

    # Create a pivot table for paper
    pivot = Article_df.pivot_table(
        values='ArtifactScore',
        index='model',
        columns='Category',
        aggfunc='count',
        fill_value=0
    )
    print("\n CONTINGENCY TABLE (for paper):")
    print(pivot)

    return


@app.cell
def _(Article_df, model):
    #Visualizations (Matplotlib)
    import os
    os.makedirs("metrics/figures", exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        # Plot 1: Scatter (SensAcc vs StabSem)
        colors = {'llama2': 'blue', 'flan_t5': 'green', 'gpt4_mini': 'red'}
        for _model in Article_df['model'].unique():
            data = Article_df[Article_df['model'] == _model]
            axes[0].scatter(data['SensAcc'], data['StabSem'], 
                           label=_model, alpha=0.6, s=100, color=colors.get(model, 'gray'))
    
        axes[0].set_xlabel('Accuracy Sensitivity (SensAcc)', fontsize=12)
        axes[0].set_ylabel('Semantic Stability (StabSem)', fontsize=12)
        axes[0].set_title('Sensitivity vs Stability', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
        # Plot 2: Bar chart (Artifact Scores by Model)
        model_avg = Article_df.groupby('model')['ArtifactScore'].mean()
        axes[1].bar(model_avg.index, model_avg.values, color=['blue', 'green', 'red'])
        axes[1].set_ylabel('Average Artifact Score', fontsize=12)
        axes[1].set_title('Artifact Score by Model', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
    
        plt.tight_layout()
        plt.savefig("metrics/figures/metrics_analysis.png", dpi=150)
        print("\n Visualization saved to metrics/figures/metrics_analysis.png")
    
    except ImportError:
        print("\n  Matplotlib not installed. Skipping visualization.")

    print("\n" + "="*70)
    print(" METRICS COMPUTATION COMPLETE")
    print("="*70)
    print("\n Next: Use these metrics in Week 3 for deeper analysis")
    print("   - Create case studies of artifacts vs real issues")
    print("   - Write up findings for paper")

    return


if __name__ == "__main__":
    app.run()
