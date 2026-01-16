from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import BytesIO
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def generate_visualizations(df, output_dir):
    """Generate all visualizations from the dataframe"""
    plots = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set style with white background
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['grid.color'] = '#e0e0e0'
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Correlation Heatmap
    if len(numeric_cols) >= 2:
        try:
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                       linewidths=0.5, linecolor="gray", square=True,
                       cbar_kws={"shrink": 0.8})
            plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            filename = f'heatmap_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(('Correlation Heatmap', filename))
        except Exception as e:
            print(f"Error generating heatmap: {e}")
    
    # 2. Distribution Histogram
    if len(numeric_cols) >= 1:
        try:
            plt.figure(figsize=(12, 6))
            col = numeric_cols[0]
            sns.histplot(df[col].dropna(), bins=30, kde=True, color='skyblue', 
                        edgecolor='black', alpha=0.7)
            mean_val = df[col].mean()
            median_val = df[col].median()
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='blue', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')
            plt.title(f'Distribution of {col}', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            filename = f'histogram_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(('Distribution Histogram', filename))
        except Exception as e:
            print(f"Error generating histogram: {e}")
    
    # 3. Box Plot
    if len(numeric_cols) >= 1:
        try:
            plt.figure(figsize=(12, 6))
            cols_to_plot = numeric_cols[:min(5, len(numeric_cols))]
            df[cols_to_plot].boxplot(patch_artist=True)
            plt.title('Box Plot - Outlier Detection', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Values', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            filename = f'boxplot_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(('Box Plot', filename))
        except Exception as e:
            print(f"Error generating boxplot: {e}")
    
    # 4. Scatter Plot
    if len(numeric_cols) >= 2:
        try:
            plt.figure(figsize=(10, 8))
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            colors = np.random.rand(len(df))
            sizes = np.abs(df[numeric_cols[0]]) * 10 if len(df) < 1000 else 50
            plt.scatter(df[x_col], df[y_col], c=colors, s=sizes, 
                       cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
            plt.colorbar(label='Color Scale')
            plt.title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'scatter_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(('Scatter Plot', filename))
        except Exception as e:
            print(f"Error generating scatter plot: {e}")
    
    # 5. Bar Chart
    if len(categorical_cols) >= 1 and len(df) < 100:
        try:
            plt.figure(figsize=(12, 6))
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(10)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(value_counts)))
            bars = plt.bar(range(len(value_counts)), value_counts.values, 
                          color=colors, edgecolor='black', linewidth=1.2)
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
            plt.title(f'Top Categories - {col}', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Count', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            filename = f'barchart_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(('Bar Chart', filename))
        except Exception as e:
            print(f"Error generating bar chart: {e}")
    
    # 6. Pie Chart
    if len(categorical_cols) >= 1:
        try:
            plt.figure(figsize=(10, 8))
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(6)
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
            explode = [0.05 if i == 0 else 0 for i in range(len(value_counts))]
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                   colors=colors, explode=explode, shadow=True, startangle=140)
            plt.title(f'Distribution - {col}', fontsize=16, fontweight='bold', pad=20)
            plt.axis('equal')
            plt.tight_layout()
            filename = f'piechart_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(('Pie Chart', filename))
        except Exception as e:
            print(f"Error generating pie chart: {e}")
    
    # 7. Line Plot
    if len(numeric_cols) >= 1:
        try:
            plt.figure(figsize=(12, 6))
            col = numeric_cols[0]
            data_sample = df[col].dropna().head(100)
            plt.plot(range(len(data_sample)), data_sample.values, 
                    marker='o', linestyle='-', linewidth=2, markersize=4, color='purple')
            plt.title(f'Line Plot - {col}', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Index', fontsize=12)
            plt.ylabel(col, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'lineplot_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(('Line Plot', filename))
        except Exception as e:
            print(f"Error generating line plot: {e}")
    
    # 8. Violin Plot
    if len(numeric_cols) >= 2:
        try:
            plt.figure(figsize=(12, 6))
            cols_to_plot = numeric_cols[:min(4, len(numeric_cols))]
            data_to_plot = [df[col].dropna().values for col in cols_to_plot]
            parts = plt.violinplot(data_to_plot, showmeans=True, showmedians=True)
            plt.xticks(range(1, len(cols_to_plot) + 1), cols_to_plot, rotation=45, ha='right')
            plt.title('Violin Plot - Distribution Comparison', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Values', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            filename = f'violin_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(('Violin Plot', filename))
        except Exception as e:
            print(f"Error generating violin plot: {e}")
    
    # 9. Area Plot
    if len(numeric_cols) >= 2:
        try:
            plt.figure(figsize=(12, 6))
            cols = numeric_cols[:min(3, len(numeric_cols))]
            data_sample = df[cols].head(50).fillna(0)
            data_sample.plot.area(alpha=0.7, stacked=True, figsize=(12, 6))
            plt.title('Area Plot - Stacked Trends', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Index', fontsize=12)
            plt.ylabel('Values', fontsize=12)
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'areaplot_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(('Area Plot', filename))
        except Exception as e:
            print(f"Error generating area plot: {e}")
    
    # 10. KDE Plot
    if len(numeric_cols) >= 1:
        try:
            plt.figure(figsize=(12, 6))
            cols = numeric_cols[:min(3, len(numeric_cols))]
            for col in cols:
                sns.kdeplot(df[col].dropna(), fill=True, alpha=0.5, linewidth=2, label=col)
            plt.title('KDE Plot - Density Distribution', fontsize=16, fontweight='bold', pad=20, color='black')
            plt.xlabel('Value', fontsize=12, color='black')
            plt.ylabel('Density', fontsize=12, color='black')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'kde_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('KDE Plot', filename))
        except Exception as e:
            print(f"Error generating KDE plot: {e}")
    
    # 11. Pairplot
    if len(numeric_cols) >= 2 and len(df) <= 1000:
        try:
            cols_to_plot = numeric_cols[:min(4, len(numeric_cols))]
            g = sns.pairplot(df[cols_to_plot].dropna(), corner=True, diag_kind='kde', 
                           plot_kws={'alpha':0.6, 's':30, 'edgecolor':'#FFD700', 'linewidth':0.5})
            g.fig.suptitle('Pair Plot - Variable Relationships', y=1.02, fontsize=16, fontweight='bold', color='black')
            filename = f'pairplot_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Pair Plot', filename))
        except Exception as e:
            print(f"Error generating pairplot: {e}")
    
    # 12. Hexbin Plot
    if len(numeric_cols) >= 2 and len(df) >= 100:
        try:
            plt.figure(figsize=(10, 8))
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            plt.hexbin(df[x_col].dropna(), df[y_col].dropna(), gridsize=30, cmap='YlOrBr', mincnt=1)
            plt.colorbar(label='Count')
            plt.title(f'Hexbin Plot: {x_col} vs {y_col}', fontsize=16, fontweight='bold', pad=20, color='black')
            plt.xlabel(x_col, fontsize=12, color='black')
            plt.ylabel(y_col, fontsize=12, color='black')
            plt.tight_layout()
            filename = f'hexbin_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Hexbin Plot', filename))
        except Exception as e:
            print(f"Error generating hexbin: {e}")
    
    # 13. Swarm Plot
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1 and len(df) <= 500:
        try:
            plt.figure(figsize=(12, 6))
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            categories = df[cat_col].value_counts().head(5).index
            data_subset = df[df[cat_col].isin(categories)]
            sns.swarmplot(data=data_subset, x=cat_col, y=num_col, palette='YlOrBr', alpha=0.7)
            plt.title(f'Swarm Plot: {num_col} by {cat_col}', fontsize=16, fontweight='bold', pad=20, color='black')
            plt.xlabel(cat_col, fontsize=12, color='black')
            plt.ylabel(num_col, fontsize=12, color='black')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            filename = f'swarm_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Swarm Plot', filename))
        except Exception as e:
            print(f"Error generating swarm plot: {e}")
    
    # 14. Ridge Plot (Multiple Distributions)
    if len(numeric_cols) >= 3:
        try:
            fig, axes = plt.subplots(len(numeric_cols[:4]), 1, figsize=(12, 10), sharex=True)
            if len(numeric_cols[:4]) == 1:
                axes = [axes]
            for idx, col in enumerate(numeric_cols[:4]):
                data = df[col].dropna()
                axes[idx].fill_between(np.linspace(data.min(), data.max(), 100),
                                      0,
                                      [np.mean(np.abs(data - x) < data.std()/10) for x in np.linspace(data.min(), data.max(), 100)],
                                      alpha=0.6, color=plt.cm.YlOrBr(idx/4))
                axes[idx].set_ylabel(col, fontsize=10, color='black')
                axes[idx].set_facecolor('white')
            plt.suptitle('Ridge Plot - Distribution Comparison', fontsize=16, fontweight='bold', color='black')
            plt.tight_layout()
            filename = f'ridge_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Ridge Plot', filename))
        except Exception as e:
            print(f"Error generating ridge plot: {e}")
    
    # 15. Count Plot
    if len(categorical_cols) >= 1:
        try:
            plt.figure(figsize=(12, 6))
            cat_col = categorical_cols[0]
            top_cats = df[cat_col].value_counts().head(10)
            sns.countplot(data=df[df[cat_col].isin(top_cats.index)], y=cat_col, 
                         palette='YlOrBr', order=top_cats.index)
            plt.title(f'Count Plot - {cat_col}', fontsize=16, fontweight='bold', pad=20, color='black')
            plt.xlabel('Count', fontsize=12, color='black')
            plt.ylabel(cat_col, fontsize=12, color='black')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            filename = f'countplot_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Count Plot', filename))
        except Exception as e:
            print(f"Error generating count plot: {e}")
    
    # 16. Joint Plot
    if len(numeric_cols) >= 2:
        try:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            g = sns.jointplot(data=df, x=x_col, y=y_col, kind='hex', color='#FFA500', marginal_kws={'color': '#FFD700'})
            g.fig.suptitle(f'Joint Plot: {x_col} vs {y_col}', y=1.02, fontsize=16, fontweight='bold', color='black')
            filename = f'jointplot_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Joint Plot', filename))
        except Exception as e:
            print(f"Error generating joint plot: {e}")
    
    # 17. Stacked Bar Chart
    if len(categorical_cols) >= 2 and len(df) < 1000:
        try:
            cat1, cat2 = categorical_cols[0], categorical_cols[1]
            top_cat1 = df[cat1].value_counts().head(5).index
            top_cat2 = df[cat2].value_counts().head(5).index
            cross_tab = pd.crosstab(df[df[cat1].isin(top_cat1)][cat1], 
                                   df[df[cat2].isin(top_cat2)][cat2])
            
            plt.figure(figsize=(12, 6))
            cross_tab.plot(kind='bar', stacked=True, colormap='YlOrBr', figsize=(12, 6))
            plt.title(f'Stacked Bar: {cat1} vs {cat2}', fontsize=16, fontweight='bold', pad=20, color='black')
            plt.xlabel(cat1, fontsize=12, color='black')
            plt.ylabel('Count', fontsize=12, color='black')
            plt.legend(title=cat2, bbox_to_anchor=(1.05, 1))
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            filename = f'stacked_bar_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Stacked Bar Chart', filename))
        except Exception as e:
            print(f"Error generating stacked bar: {e}")
    
    # 18. Regression Plot
    if len(numeric_cols) >= 2:
        try:
            plt.figure(figsize=(10, 8))
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={'alpha':0.5, 'color':'#FFD700'}, 
                       line_kws={'color':'#FFA500', 'linewidth':3})
            plt.title(f'Regression Plot: {x_col} vs {y_col}', fontsize=16, fontweight='bold', pad=20, color='black')
            plt.xlabel(x_col, fontsize=12, color='black')
            plt.ylabel(y_col, fontsize=12, color='black')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'regplot_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Regression Plot', filename))
        except Exception as e:
            print(f"Error generating regression plot: {e}")
    
    # 19. 3D Scatter Plot
    if len(numeric_cols) >= 3:
        try:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            x_col, y_col, z_col = numeric_cols[0], numeric_cols[1], numeric_cols[2]
            colors = np.linspace(0, 1, len(df))
            scatter = ax.scatter(df[x_col], df[y_col], df[z_col], c=colors, 
                               cmap='YlOrBr', s=50, alpha=0.6, edgecolor='#FFD700', linewidth=0.5)
            
            ax.set_xlabel(x_col, fontsize=12, color='black')
            ax.set_ylabel(y_col, fontsize=12, color='black')
            ax.set_zlabel(z_col, fontsize=12, color='black')
            ax.set_title(f'3D Scatter: {x_col}, {y_col}, {z_col}', fontsize=16, fontweight='bold', pad=20, color='black')
            ax.set_facecolor('white')
            fig.colorbar(scatter, ax=ax, label='Index', pad=0.1)
            plt.tight_layout()
            filename = f'scatter3d_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('3D Scatter Plot', filename))
        except Exception as e:
            print(f"Error generating 3D scatter: {e}")
    
    # 20. Strip Plot
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        try:
            plt.figure(figsize=(12, 6))
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            categories = df[cat_col].value_counts().head(8).index
            data_subset = df[df[cat_col].isin(categories)]
            sns.stripplot(data=data_subset, x=cat_col, y=num_col, palette='YlOrBr', 
                         jitter=0.3, alpha=0.6, size=5)
            plt.title(f'Strip Plot: {num_col} by {cat_col}', fontsize=16, fontweight='bold', pad=20, color='black')
            plt.xlabel(cat_col, fontsize=12, color='black')
            plt.ylabel(num_col, fontsize=12, color='black')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            filename = f'stripplot_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Strip Plot', filename))
        except Exception as e:
            print(f"Error generating strip plot: {e}")
    
    # 21. Point Plot
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        try:
            plt.figure(figsize=(12, 6))
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            categories = df[cat_col].value_counts().head(10).index
            data_subset = df[df[cat_col].isin(categories)]
            sns.pointplot(data=data_subset, x=cat_col, y=num_col, color='#FFD700', 
                         markers='D', linestyles='--', ci=95)
            plt.title(f'Point Plot: {num_col} by {cat_col}', fontsize=16, fontweight='bold', pad=20, color='black')
            plt.xlabel(cat_col, fontsize=12, color='black')
            plt.ylabel(num_col, fontsize=12, color='black')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'pointplot_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Point Plot', filename))
        except Exception as e:
            print(f"Error generating point plot: {e}")
    
    # 22. Clustermap
    if len(numeric_cols) >= 3 and len(df) <= 100:
        try:
            cols = numeric_cols[:min(6, len(numeric_cols))]
            corr_data = df[cols].dropna().corr()
            g = sns.clustermap(corr_data, cmap='YlOrBr', annot=True, fmt='.2f', 
                              figsize=(10, 10), linewidths=0.5, cbar_kws={'label': 'Correlation'})
            g.fig.suptitle('Cluster Map - Hierarchical Clustering', y=0.98, fontsize=16, fontweight='bold', color='black')
            filename = f'clustermap_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Cluster Map', filename))
        except Exception as e:
            print(f"Error generating clustermap: {e}")
    
    # 23. ECDF Plot
    if len(numeric_cols) >= 1:
        try:
            plt.figure(figsize=(12, 6))
            cols = numeric_cols[:min(3, len(numeric_cols))]
            for col in cols:
                sns.ecdfplot(data=df, x=col, label=col, linewidth=2)
            plt.title('ECDF Plot - Cumulative Distribution', fontsize=16, fontweight='bold', pad=20, color='black')
            plt.xlabel('Value', fontsize=12, color='black')
            plt.ylabel('ECDF', fontsize=12, color='black')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'ecdf_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('ECDF Plot', filename))
        except Exception as e:
            print(f"Error generating ECDF: {e}")
    
    # 24. Donut Chart
    if len(categorical_cols) >= 1:
        try:
            plt.figure(figsize=(10, 8))
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(8)
            colors = plt.cm.YlOrBr(np.linspace(0.3, 0.9, len(value_counts)))
            
            wedges, texts, autotexts = plt.pie(value_counts.values, labels=value_counts.index, 
                                               autopct='%1.1f%%', colors=colors, startangle=140,
                                               wedgeprops=dict(width=0.4, edgecolor='#FFD700'))
            
            for text in texts:
                text.set_color('#FFD700')
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
            
            plt.title(f'Donut Chart - {col}', fontsize=16, fontweight='bold', pad=20, color='black')
            plt.tight_layout()
            filename = f'donut_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(('Donut Chart', filename))
        except Exception as e:
            print(f"Error generating donut chart: {e}")
    
    # 25. Heatmap with Annotations (missing values)
    if len(df.columns) <= 30:
        try:
            plt.figure(figsize=(12, 8))
            missing = df.isnull()
            if missing.sum().sum() > 0:
                sns.heatmap(missing, cbar=True, cmap='YlOrBr', yticklabels=False, 
                           cbar_kws={'label': 'Missing'})
                plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold', pad=20, color='black')
                plt.xlabel('Columns', fontsize=12, color='black')
                plt.ylabel('Rows', fontsize=12, color='black')
                plt.tight_layout()
                filename = f'missing_heatmap_{timestamp}.png'
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                plots.append(('Missing Values Heatmap', filename))
        except Exception as e:
            print(f"Error generating missing heatmap: {e}")
    
    return plots

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    try:
        # Read CSV
        df = pd.read_csv(file)
        
        # Generate visualizations
        plots = generate_visualizations(df, app.config['OUTPUT_FOLDER'])
        
        # Get basic stats
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns),
            'missing_values': int(df.isnull().sum().sum())
        }
        
        return jsonify({
            'success': True,
            'plots': plots,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)