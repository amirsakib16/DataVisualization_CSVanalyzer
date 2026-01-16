from flask import Flask, render_template, request, send_file, jsonify
import os
import sys

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Now import other libraries
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set environment variables for matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
plt.ioff()  # Turn off interactive mode

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
OUTPUT_DIR = os.path.join(STATIC_DIR, 'outputs')
TEMP_DIR = '/tmp/matplotlib'

# Create necessary directories with error handling
for directory in [STATIC_DIR, OUTPUT_DIR, TEMP_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
        os.chmod(directory, 0o777)  # Ensure write permissions
    except Exception as e:
        print(f"Warning: Could not create {directory}: {e}")

def generate_visualizations(df, output_dir):
    """Generate all visualizations from the dataframe"""
    plots = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Force matplotlib to use Agg backend
        plt.switch_backend('Agg')
        
        # Set style with white background
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['text.color'] = 'black'
        plt.rcParams['axes.labelcolor'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        plt.rcParams['grid.color'] = '#e0e0e0'
        plt.rcParams['font.size'] = 10
        
    except Exception as e:
        print(f"Error setting matplotlib config: {e}")
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Found {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")
    
    # 1. Correlation Heatmap
    if len(numeric_cols) >= 2:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                       linewidths=0.5, linecolor="gray", square=True,
                       cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            filename = f'heatmap_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            plots.append(('Correlation Heatmap', filename))
            print(f"Generated: {filename}")
        except Exception as e:
            print(f"Error generating heatmap: {e}")
    
    # 2. Distribution Histogram
    if len(numeric_cols) >= 1:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            col = numeric_cols[0]
            sns.histplot(df[col].dropna(), bins=30, kde=True, color='skyblue', 
                        edgecolor='black', alpha=0.7, ax=ax)
            mean_val = df[col].mean()
            median_val = df[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='blue', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.set_title(f'Distribution of {col}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            filename = f'histogram_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            plots.append(('Distribution Histogram', filename))
            print(f"Generated: {filename}")
        except Exception as e:
            print(f"Error generating histogram: {e}")
    
    # 3. Box Plot
    if len(numeric_cols) >= 1:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            cols_to_plot = numeric_cols[:min(5, len(numeric_cols))]
            df[cols_to_plot].boxplot(patch_artist=True, ax=ax)
            ax.set_title('Box Plot - Outlier Detection', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Values', fontsize=12)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            filename = f'boxplot_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            plots.append(('Box Plot', filename))
            print(f"Generated: {filename}")
        except Exception as e:
            print(f"Error generating boxplot: {e}")
    
    # 4. Scatter Plot
    if len(numeric_cols) >= 2:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            colors = np.random.rand(len(df))
            sizes = 50
            scatter = ax.scatter(df[x_col], df[y_col], c=colors, s=sizes, 
                       cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Color Scale')
            ax.set_title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'scatter_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            plots.append(('Scatter Plot', filename))
            print(f"Generated: {filename}")
        except Exception as e:
            print(f"Error generating scatter plot: {e}")
    
    # 5. Bar Chart
    if len(categorical_cols) >= 1:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(10)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(value_counts)))
            bars = ax.bar(range(len(value_counts)), value_counts.values, 
                          color=colors, edgecolor='black', linewidth=1.2)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_title(f'Top Categories - {col}', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Count', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            filename = f'barchart_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            plots.append(('Bar Chart', filename))
            print(f"Generated: {filename}")
        except Exception as e:
            print(f"Error generating bar chart: {e}")
    
    # 6. Line Plot
    if len(numeric_cols) >= 1:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            col = numeric_cols[0]
            data_sample = df[col].dropna().head(100)
            ax.plot(range(len(data_sample)), data_sample.values, 
                    marker='o', linestyle='-', linewidth=2, markersize=4, color='purple')
            ax.set_title(f'Line Plot - {col}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Index', fontsize=12)
            ax.set_ylabel(col, fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'lineplot_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            plots.append(('Line Plot', filename))
            print(f"Generated: {filename}")
        except Exception as e:
            print(f"Error generating line plot: {e}")
    
    # 7. Violin Plot
    if len(numeric_cols) >= 2:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            cols_to_plot = numeric_cols[:min(4, len(numeric_cols))]
            data_to_plot = [df[col].dropna().values for col in cols_to_plot]
            parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(cols_to_plot) + 1))
            ax.set_xticklabels(cols_to_plot, rotation=45, ha='right')
            ax.set_title('Violin Plot - Distribution Comparison', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Values', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            filename = f'violin_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            plots.append(('Violin Plot', filename))
            print(f"Generated: {filename}")
        except Exception as e:
            print(f"Error generating violin plot: {e}")
    
    # 8. KDE Plot
    if len(numeric_cols) >= 1:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            cols = numeric_cols[:min(3, len(numeric_cols))]
            for col in cols:
                sns.kdeplot(df[col].dropna(), fill=True, alpha=0.5, linewidth=2, label=col, ax=ax)
            ax.set_title('KDE Plot - Density Distribution', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Value', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'kde_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            plots.append(('KDE Plot', filename))
            print(f"Generated: {filename}")
        except Exception as e:
            print(f"Error generating KDE plot: {e}")
    
    print(f"Successfully generated {len(plots)} plots")
    return plots

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("Starting analysis...")
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files are allowed'}), 400
        
        # Read CSV
        print("Reading CSV file...")
        df = pd.read_csv(file)
        print(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Validate dataframe
        if df.empty:
            return jsonify({'success': False, 'error': 'CSV file is empty'}), 400
        
        # Generate visualizations
        print("Generating visualizations...")
        plots = generate_visualizations(df, OUTPUT_DIR)
        print(f"Generated {len(plots)} plots successfully")
        
        # Get basic stats
        stats = {
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'numeric_cols': int(len(df.select_dtypes(include=[np.number]).columns)),
            'categorical_cols': int(len(df.select_dtypes(include=['object', 'category']).columns)),
            'missing_values': int(df.isnull().sum().sum())
        }
        
        print(f"Returning response with {len(plots)} plots")
        return jsonify({
            'success': True,
            'plots': plots,
            'stats': stats
        })
    
    except Exception as e:
        print(f"Error in analyze route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/download/<filename>')
def download(filename):
    try:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("DataViz Pro - Starting Server")
    print("="*60)
    print(f"Python version: {sys.version}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Static directory: {STATIC_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Temp directory: {TEMP_DIR}")
    print("="*60)
    
    # Verify directories exist and are writable
    for dir_name, dir_path in [("Static", STATIC_DIR), ("Output", OUTPUT_DIR), ("Temp", TEMP_DIR)]:
        if os.path.exists(dir_path):
            print(f"✓ {dir_name} directory exists: {dir_path}")
            if os.access(dir_path, os.W_OK):
                print(f"✓ {dir_name} directory is writable")
            else:
                print(f"✗ WARNING: {dir_name} directory is NOT writable")
        else:
            print(f"✗ WARNING: {dir_name} directory does NOT exist")
    
    print("="*60)
    app.run(debug=True, port=5000, host='0.0.0.0')