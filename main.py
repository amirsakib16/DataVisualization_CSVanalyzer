from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def set_dark_style():
    """Set dark theme for matplotlib"""
    plt.style.use('dark_background')
    sns.set_palette("husl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Basic info
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Generate visualizations
        graphs = generate_all_graphs(df)
        
        return jsonify({
            'success': True,
            'info': info,
            'graphs': graphs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_all_graphs(df):
    """Generate all available graphs based on data"""
    graphs = []
    set_dark_style()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 1. BASIC LINE PLOT
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        for col in numeric_cols[:5]:
            ax.plot(df.index, df[col], marker='o', linewidth=2, markersize=4, alpha=0.8, label=col)
        ax.set_title('Line Plot - Numeric Trends', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_xlabel('Index', fontsize=12, color='#e94560')
        ax.set_ylabel('Values', fontsize=12, color='#e94560')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'Line Plot', 'image': fig_to_base64(fig)})
    
    # 2. ADVANCED LINE PLOT with annotations
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        col = numeric_cols[0]
        ax.plot(df.index, df[col], marker='o', linestyle='--', color='m', linewidth=2, markersize=8)
        for i in range(0, len(df), max(1, len(df)//10)):
            ax.annotate(f"{df[col].iloc[i]:.1f}", (df.index[i], df[col].iloc[i] + 0.5), 
                       ha='center', fontsize=9, color='white')
        ax.set_title(f'Advanced Line Plot - {col}', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.grid(True, linestyle='--', alpha=0.5)
        graphs.append({'name': 'Advanced Line Plot', 'image': fig_to_base64(fig)})
    
    # 3. BAR PLOT
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        bars = ax.bar(grouped.index, grouped.values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(grouped))), 
                      edgecolor='white', linewidth=1.5)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                   ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')
        ax.set_title(f'Bar Plot: {cat_col} vs {num_col}', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_xlabel(cat_col, fontsize=12, color='#e94560')
        ax.set_ylabel(f'Mean {num_col}', fontsize=12, color='#e94560')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Bar Plot', 'image': fig_to_base64(fig)})
    
    # 4. HORIZONTAL BAR PLOT
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=True).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        bars = ax.barh(grouped.index, grouped.values, color=plt.cm.plasma(np.linspace(0.2, 0.8, len(grouped))), 
                       edgecolor='black', height=0.6)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                   va='center', fontsize=10, color='white', fontweight='bold')
        ax.set_title(f'Horizontal Bar Plot: {cat_col} vs {num_col}', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Horizontal Bar Plot', 'image': fig_to_base64(fig)})
    
    # 5. SCATTER PLOT
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        scatter = ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], 
                           c=df[numeric_cols[0]], s=100, cmap='viridis', alpha=0.7, 
                           edgecolors='white', linewidth=0.5)
        ax.set_title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}', 
                    fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_xlabel(numeric_cols[0], fontsize=12, color='#e94560')
        ax.set_ylabel(numeric_cols[1], fontsize=12, color='#e94560')
        plt.colorbar(scatter, ax=ax, label='Intensity')
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'Scatter Plot', 'image': fig_to_base64(fig)})
    
    # 6. ADVANCED SCATTER PLOT with size encoding
    if len(numeric_cols) >= 3:
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sizes = (df[numeric_cols[2]] - df[numeric_cols[2]].min() + 1) * 100
        scatter = ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], 
                           c=df[numeric_cols[0]] + df[numeric_cols[1]], s=sizes, 
                           cmap='plasma', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_title(f'Advanced Scatter: Size={numeric_cols[2]}', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_xlabel(numeric_cols[0], fontsize=12, color='#e94560')
        ax.set_ylabel(numeric_cols[1], fontsize=12, color='#e94560')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Color Intensity')
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'Advanced Scatter Plot', 'image': fig_to_base64(fig)})
    
    # 7. HISTOGRAM
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.hist(df[numeric_cols[0]].dropna(), bins=30, color='#00d4ff', 
               edgecolor='white', alpha=0.8)
        mean_val = df[numeric_cols[0]].mean()
        median_val = df[numeric_cols[0]].median()
        ax.axvline(mean_val, color='#e94560', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='#9d4edd', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.set_title(f'Histogram: {numeric_cols[0]}', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_xlabel('Value', fontsize=12, color='#e94560')
        ax.set_ylabel('Frequency', fontsize=12, color='#e94560')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Histogram', 'image': fig_to_base64(fig)})
    
    # 8. ADVANCED HISTOGRAM with KDE
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.histplot(df[numeric_cols[0]].dropna(), bins=30, kde=True, color='mediumseagreen', 
                    edgecolor='black', alpha=0.7, ax=ax)
        ax.set_title(f'Advanced Histogram with KDE - {numeric_cols[0]}', 
                    fontsize=16, fontweight='bold', color='#00d4ff')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Histogram with KDE', 'image': fig_to_base64(fig)})
    
    # 9. PIE CHART
    if len(categorical_cols) >= 1:
        cat_col = categorical_cols[0]
        value_counts = df[cat_col].value_counts().head(8)
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        explode = [0.05 if i == 0 else 0 for i in range(len(value_counts))]
        wedges, texts, autotexts = ax.pie(value_counts, labels=value_counts.index, 
                                          autopct='%1.1f%%', colors=colors, explode=explode,
                                          startangle=90, shadow=True,
                                          textprops={'fontsize': 11, 'weight': 'bold'})
        ax.set_title(f'Pie Chart: {cat_col} Distribution', fontsize=16, fontweight='bold', color='#00d4ff')
        graphs.append({'name': 'Pie Chart', 'image': fig_to_base64(fig)})
    
    # 10. DONUT CHART
    if len(categorical_cols) >= 1:
        cat_col = categorical_cols[0]
        value_counts = df[cat_col].value_counts().head(8)
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(value_counts)))
        wedges, texts, autotexts = ax.pie(value_counts, labels=value_counts.index, 
                                          autopct='%1.1f%%', colors=colors,
                                          startangle=140, wedgeprops=dict(width=0.3, edgecolor='w'))
        ax.set_title(f'Donut Chart: {cat_col}', fontsize=16, fontweight='bold', color='#00d4ff')
        graphs.append({'name': 'Donut Chart', 'image': fig_to_base64(fig)})
    
    # 11. BOX PLOT
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.boxplot(data=df[numeric_cols[:5]], palette='Set2', ax=ax, linewidth=2)
        ax.set_title('Box Plot - Distribution Comparison', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_ylabel('Values', fontsize=12, color='#e94560')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Box Plot', 'image': fig_to_base64(fig)})
    
    # 12. ADVANCED BOX PLOT with swarm overlay
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        sample_df = df.sample(n=min(500, len(df)))
        
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.boxplot(x=cat_col, y=num_col, data=sample_df, palette='Set2', ax=ax, linewidth=2, notch=True)
        sns.swarmplot(x=cat_col, y=num_col, data=sample_df, color='k', alpha=0.4, size=3, ax=ax)
        ax.set_title(f'Advanced Box Plot with Swarm: {cat_col} vs {num_col}', 
                    fontsize=16, fontweight='bold', color='#00d4ff')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Box Plot with Swarm', 'image': fig_to_base64(fig)})
    
    # 13. VIOLIN PLOT
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.violinplot(x=cat_col, y=num_col, data=df, palette='mako', ax=ax, inner='quartile')
        ax.set_title(f'Violin Plot: {cat_col} vs {num_col}', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_xlabel(cat_col, fontsize=12, color='#e94560')
        ax.set_ylabel(num_col, fontsize=12, color='#e94560')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Violin Plot', 'image': fig_to_base64(fig)})
    
    # 14. ADVANCED VIOLIN with split
    if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
        cat_col1 = categorical_cols[0]
        cat_col2 = categorical_cols[1]
        num_col = numeric_cols[0]
        
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.violinplot(x=cat_col1, y=num_col, hue=cat_col2, data=df, split=True, 
                      palette='coolwarm', ax=ax, inner='quart')
        ax.set_title(f'Split Violin Plot: {cat_col1} vs {num_col} by {cat_col2}', 
                    fontsize=16, fontweight='bold', color='#00d4ff')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Split Violin Plot', 'image': fig_to_base64(fig)})
    
    # 15. STRIP PLOT
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        sample_df = df.sample(n=min(500, len(df)))
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.stripplot(x=cat_col, y=num_col, data=sample_df, jitter=0.3, 
                     size=6, palette='Set1', alpha=0.7, ax=ax)
        ax.set_title(f'Strip Plot: {cat_col} vs {num_col}', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Strip Plot', 'image': fig_to_base64(fig)})
    
    # 16. SWARM PLOT
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        sample_df = df.sample(n=min(300, len(df)))
        
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.swarmplot(x=cat_col, y=num_col, data=sample_df, size=6, 
                     palette='Set2', alpha=0.8, ax=ax)
        ax.set_title(f'Swarm Plot: {cat_col} vs {num_col}', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Swarm Plot', 'image': fig_to_base64(fig)})
    
    # 17. POINT PLOT
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.pointplot(x=cat_col, y=num_col, data=df, ax=ax, markers='D', 
                     linestyles='--', palette='Dark2', ci=95)
        ax.set_title(f'Point Plot: {cat_col} vs {num_col}', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'Point Plot', 'image': fig_to_base64(fig)})
    
    # 18. COUNT PLOT
    if len(categorical_cols) >= 1:
        cat_col = categorical_cols[0]
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        order = df[cat_col].value_counts().head(10).index
        sns.countplot(y=cat_col, data=df, palette='viridis', ax=ax, order=order)
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.5, p.get_y() + p.get_height()/2, int(width),
                   ha='left', va='center', fontsize=10, fontweight='bold', color='white')
        ax.set_title(f'Count Plot: {cat_col}', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_xlabel('Count', fontsize=12, color='#e94560')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Count Plot', 'image': fig_to_base64(fig)})
    
    # 19. AREA PLOT
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        for col in numeric_cols[:3]:
            ax.fill_between(df.index, df[col], alpha=0.5, label=col)
            ax.plot(df.index, df[col], linewidth=2)
        ax.set_title('Area Plot - Trends', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_xlabel('Index', fontsize=12, color='#e94560')
        ax.set_ylabel('Values', fontsize=12, color='#e94560')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'Area Plot', 'image': fig_to_base64(fig)})
    
    # 20. STACKED AREA PLOT
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        cols = numeric_cols[:3]
        colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(cols)))
        ax.stackplot(df.index, *[df[col] for col in cols], labels=cols, 
                    colors=colors, alpha=0.85)
        ax.set_title('Stacked Area Plot', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.legend(loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        graphs.append({'name': 'Stacked Area Plot', 'image': fig_to_base64(fig)})
    
    # 21. STEM PLOT
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        x = df.index[:30]
        y = df[numeric_cols[0]].iloc[:30]
        markerline, stemlines, baseline = ax.stem(x, y, linefmt='C1-', markerfmt='o', basefmt='C3-')
        plt.setp(markerline, markersize=10, markerfacecolor='orange', markeredgecolor='red')
        plt.setp(stemlines, linewidth=2, linestyle='--', color='purple')
        ax.set_title(f'Stem Plot: {numeric_cols[0]}', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'Stem Plot', 'image': fig_to_base64(fig)})
    
    # 22. STEP PLOT
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        x = df.index[:50]
        ax.step(x, df[numeric_cols[0]].iloc[:50], where='pre', label=numeric_cols[0], linewidth=2, color='blue', marker='o')
        ax.step(x, df[numeric_cols[1]].iloc[:50], where='post', label=numeric_cols[1], linewidth=2, color='green', marker='s')
        ax.set_title('Step Plot', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'Step Plot', 'image': fig_to_base64(fig)})
    
    # 23. KDE PLOT 1D
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        for col in numeric_cols[:3]:
            sns.kdeplot(df[col].dropna(), fill=True, alpha=0.5, linewidth=2.5, label=col, ax=ax)
        ax.set_title('KDE Plot 1D - Density Estimation', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_xlabel('Value', fontsize=12, color='#e94560')
        ax.set_ylabel('Density', fontsize=12, color='#e94560')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'KDE Plot 1D', 'image': fig_to_base64(fig)})
    
    # 24. KDE PLOT 2D
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.kdeplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]], fill=True, 
                   cmap='viridis', levels=15, thresh=0.05, ax=ax)
        ax.set_title(f'2D KDE Plot: {numeric_cols[0]} vs {numeric_cols[1]}', 
                    fontsize=16, fontweight='bold', color='#00d4ff')
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'KDE Plot 2D', 'image': fig_to_base64(fig)})
    
    # 25. ECDF PLOT
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        for col in numeric_cols[:3]:
            sns.ecdfplot(data=df[col].dropna(), label=col, linewidth=2, ax=ax)
        ax.set_title('ECDF Plot - Cumulative Distribution', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'ECDF Plot', 'image': fig_to_base64(fig)})
    
    # 26. RUG PLOT
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.kdeplot(df[numeric_cols[0]].dropna(), fill=True, alpha=0.3, linewidth=2, color='purple', ax=ax)
        sns.rugplot(df[numeric_cols[0]].dropna(), height=0.2, linewidth=1.5, color='black', ax=ax)
        ax.set_title(f'Rug Plot with KDE: {numeric_cols[0]}', fontsize=16, fontweight='bold', color='#00d4ff')
        graphs.append({'name': 'Rug Plot', 'image': fig_to_base64(fig)})
    
    # 27. JOINT PLOT (Marginal Histogram)
    if len(numeric_cols) >= 2:
        g = sns.JointGrid(data=df, x=numeric_cols[0], y=numeric_cols[1], height=8)
        g.plot_joint(sns.scatterplot, hue=None, s=80, alpha=0.6, edgecolor='black')
        g.plot_marginals(sns.histplot, kde=True, bins=30)
        g.fig.patch.set_facecolor('#1a1a2e')
        g.ax_joint.set_facecolor('#16213e')
        g.ax_marg_x.set_facecolor('#16213e')
        g.ax_marg_y.set_facecolor('#16213e')
        g.fig.suptitle(f'Joint Plot: {numeric_cols[0]} vs {numeric_cols[1]}', 
                      fontsize=16, color='#00d4ff', y=0.98)
        graphs.append({'name': 'Joint Plot', 'image': fig_to_base64(g.fig)})
    
    # 28. HEXBIN PLOT
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        hb = ax.hexbin(df[numeric_cols[0]], df[numeric_cols[1]], 
                      gridsize=40, cmap='inferno', linewidths=0.8, mincnt=1)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Count')
        ax.set_title(f'Hexbin Plot: {numeric_cols[0]} vs {numeric_cols[1]}', 
                    fontsize=16, fontweight='bold', color='#00d4ff')
        ax.grid(True, linestyle=':', alpha=0.3)
        graphs.append({'name': 'Hexbin Plot', 'image': fig_to_base64(fig)})
    
    # 29. 2D HISTOGRAM
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        h = ax.hist2d(df[numeric_cols[0]].dropna(), df[numeric_cols[1]].dropna(), 
                     bins=30, cmap='Blues')
        plt.colorbar(h[3], ax=ax, label='Count')
        ax.set_title(f'2D Histogram: {numeric_cols[0]} vs {numeric_cols[1]}', 
                    fontsize=16, fontweight='bold', color='#00d4ff')
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': '2D Histogram', 'image': fig_to_base64(fig)})
    
    # 30. CORRELATION HEATMAP
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='#1a1a2e')
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   linewidths=0.5, linecolor='gray', ax=ax, cbar_kws={'label': 'Correlation'},
                   annot_kws={'size': 10, 'weight': 'bold'}, square=True)
        ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold', color='#00d4ff')
        graphs.append({'name': 'Correlation Heatmap', 'image': fig_to_base64(fig)})
    
    # 31. CLUSTERMAP
    if len(numeric_cols) >= 3 and len(df) <= 100:
        sample_data = df[numeric_cols[:5]].dropna()
        if len(sample_data) > 0:
            g = sns.clustermap(sample_data.T, cmap='vlag', figsize=(10, 8),
                             linewidths=0.8, linecolor='black', dendrogram_ratio=0.2)
            g.fig.patch.set_facecolor('#1a1a2e')
            g.fig.suptitle('Clustermap - Hierarchical Clustering', 
                          fontsize=16, color='#00d4ff', y=0.98)
            graphs.append({'name': 'Clustermap', 'image': fig_to_base64(g.fig)})
    
    # 32. PAIR PLOT
    if len(numeric_cols) >= 3 and len(df) <= 500:
        sample_df = df[numeric_cols[:4]].sample(n=min(200, len(df)))
        g = sns.pairplot(sample_df, diag_kind='kde', plot_kws={'alpha':0.6, 's':50})
        g.fig.patch.set_facecolor('#1a1a2e')
        for ax in g.axes.flatten():
            ax.set_facecolor('#16213e')
        g.fig.suptitle('Pair Plot - Relationships', fontsize=16, color='#00d4ff', y=1.02)
        graphs.append({'name': 'Pair Plot', 'image': fig_to_base64(g.fig)})
    
    # 33. BUBBLE PLOT
    if len(numeric_cols) >= 3:
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sizes = (df[numeric_cols[2]] - df[numeric_cols[2]].min() + 1) * 50
        scatter = ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], 
                           s=sizes, c=df[numeric_cols[2]], cmap='viridis', 
                           alpha=0.6, edgecolors='black', linewidth=1.2)
        ax.set_title(f'Bubble Plot: Size={numeric_cols[2]}', 
                    fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_xlabel(numeric_cols[0], fontsize=12, color='#e94560')
        ax.set_ylabel(numeric_cols[1], fontsize=12, color='#e94560')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(numeric_cols[2])
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'Bubble Plot', 'image': fig_to_base64(fig)})
    
    # 34. 3D SCATTER PLOT
    if len(numeric_cols) >= 3:
        fig = plt.figure(figsize=(12, 9), facecolor='#1a1a2e')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#16213e')
        scatter = ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], df[numeric_cols[2]],
                           c=df[numeric_cols[0]], s=100, cmap='plasma', alpha=0.7, 
                           edgecolors='black', linewidth=0.5)
        ax.set_title(f'3D Scatter Plot', fontsize=16, fontweight='bold', color='#00d4ff')
        ax.set_xlabel(numeric_cols[0], color='#e94560')
        ax.set_ylabel(numeric_cols[1], color='#e94560')
        ax.set_zlabel(numeric_cols[2], color='#e94560')
        fig.colorbar(scatter, ax=ax, shrink=0.5, label='Color scale')
        graphs.append({'name': '3D Scatter Plot', 'image': fig_to_base64(fig)})
    
    # 35. REGRESSION PLOT
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.regplot(x=numeric_cols[0], y=numeric_cols[1], data=df, 
                   scatter_kws={'s': 70, 'alpha': 0.6, 'edgecolor': 'k', 'color': 'dodgerblue'},
                   line_kws={'color': 'orange', 'linewidth': 3, 'linestyle': '--'}, ax=ax)
        ax.set_title(f'Regression Plot: {numeric_cols[0]} vs {numeric_cols[1]}', 
                    fontsize=16, fontweight='bold', color='#00d4ff')
        ax.grid(True, linestyle='--', alpha=0.3)
        graphs.append({'name': 'Regression Plot', 'image': fig_to_base64(fig)})
    
    # 36. RESIDUAL PLOT
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        sns.residplot(x=numeric_cols[0], y=numeric_cols[1], data=df,
                     lowess=True, color='darkgreen',
                     scatter_kws={'s': 60, 'alpha': 0.6},
                     line_kws={'color': 'red', 'lw': 2, 'ls': '--'}, ax=ax)
        ax.set_title(f'Residual Plot: {numeric_cols[0]} vs {numeric_cols[1]}', 
                    fontsize=16, fontweight='bold', color='#00d4ff')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axhline(0, color='white', linestyle='-', linewidth=1)
        graphs.append({'name': 'Residual Plot', 'image': fig_to_base64(fig)})
    
    # 37. CUMULATIVE HISTOGRAM
    if len(numeric_cols) >= 1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.hist(df[numeric_cols[0]].dropna(), bins=30, cumulative=True, 
               color='teal', edgecolor='black', alpha=0.8)
        ax.axvline(df[numeric_cols[0]].mean(), color='red', linestyle='--', 
                  label='Mean', linewidth=2)
        ax.set_title(f'Cumulative Histogram: {numeric_cols[0]}', 
                    fontsize=16, fontweight='bold', color='#00d4ff')
        ax.legend()
        ax.grid(True, alpha=0.3)
        graphs.append({'name': 'Cumulative Histogram', 'image': fig_to_base64(fig)})
    
    # 38. HEATMAP (if categorical and numeric)
    if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
        pivot = df.pivot_table(values=numeric_cols[0], 
                              index=categorical_cols[0], 
                              columns=categorical_cols[1], 
                              aggfunc='mean')
        if pivot.shape[0] > 0 and pivot.shape[1] > 0:
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu',
                       linewidths=0.8, linecolor='black', ax=ax,
                       cbar_kws={'label': f'Mean {numeric_cols[0]}'})
            ax.set_title(f'Heatmap: {categorical_cols[0]} vs {categorical_cols[1]}', 
                        fontsize=16, fontweight='bold', color='#00d4ff')
            graphs.append({'name': 'Pivot Heatmap', 'image': fig_to_base64(fig)})
    
    # 39. FACET GRID
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 2 and len(df[categorical_cols[0]].unique()) <= 4:
        g = sns.FacetGrid(df, col=categorical_cols[0], height=4, aspect=1.2, col_wrap=2)
        g.map(sns.scatterplot, numeric_cols[0], numeric_cols[1], alpha=0.7)
        g.fig.patch.set_facecolor('#1a1a2e')
        for ax in g.axes.flatten():
            ax.set_facecolor('#16213e')
            ax.grid(True, linestyle='--', alpha=0.3)
        g.fig.suptitle(f'Facet Grid by {categorical_cols[0]}', 
                      fontsize=16, color='#00d4ff', y=1.02)
        graphs.append({'name': 'Facet Grid', 'image': fig_to_base64(g.fig)})
    
    # 40. GROUPED BAR PLOT
    if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
        cat_col1 = categorical_cols[0]
        cat_col2 = categorical_cols[1]
        num_col = numeric_cols[0]
        
        pivot = df.groupby([cat_col1, cat_col2])[num_col].mean().unstack()
        if pivot.shape[0] <= 10 and pivot.shape[1] <= 5:
            fig, ax = plt.subplots(figsize=(14, 7), facecolor='#1a1a2e')
            ax.set_facecolor('#16213e')
            pivot.plot(kind='bar', ax=ax, edgecolor='black', linewidth=1.2)
            ax.set_title(f'Grouped Bar Plot: {cat_col1} by {cat_col2}', 
                        fontsize=16, fontweight='bold', color='#00d4ff')
            ax.set_xlabel(cat_col1, fontsize=12, color='#e94560')
            ax.set_ylabel(f'Mean {num_col}', fontsize=12, color='#e94560')
            ax.legend(title=cat_col2)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            graphs.append({'name': 'Grouped Bar Plot', 'image': fig_to_base64(fig)})
    
    return graphs

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)