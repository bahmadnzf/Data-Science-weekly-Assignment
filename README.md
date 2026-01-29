# Data Science Workflow Repository

A comprehensive collection of data science tasks and methodologies, demonstrating end-to-end workflows from raw data ingestion to insightful analysis and visualization. This repository serves as both an educational resource and a practical reference for data science best practices.

## Overview

This repository provides systematic implementations of fundamental data science activities that form the foundation of any analytical project. Each component is designed to showcase professional approaches to handling, analyzing, and deriving insights from data across various domains and dataset types.

Data science is an iterative process that transforms raw, messy data into actionable insights through cleaning, exploration, statistical analysis, and visualization. This repository demonstrates the complete workflow, emphasizing reproducibility, code quality, and analytical rigor at every stage.


## Data Science Workflow Components

This repository covers all essential phases of the data science lifecycle:

### 1. Data Loading

**Purpose**: Efficiently import data from various sources into pandas DataFrames or appropriate data structures for analysis.

**Implemented Tasks:**

| Data Source | Method | Use Case | Key Considerations |
|-------------|--------|----------|-------------------|
| CSV/Excel Files | `pd.read_csv()`, `pd.read_excel()` | Most common structured data format | Handle encoding, delimiters, headers |
| Databases (SQL) | SQLAlchemy, `pd.read_sql()` | Enterprise data warehouses | Connection management, query optimization |
| JSON/XML | `pd.read_json()`, custom parsers | API responses, nested data | Normalize nested structures |
| APIs | `requests`, `urllib` | Real-time data, web services | Authentication, rate limiting, pagination |
| Web Scraping | BeautifulSoup, Selenium | Public web data | Respect robots.txt, ethical scraping |
| Parquet/HDF5 | `pd.read_parquet()`, `pd.read_hdf()` | Large datasets, optimized storage | Columnar efficiency, compression |
| Cloud Storage | boto3 (S3), Google Cloud Storage | Scalable data storage | Credentials, streaming for large files |

**Key Features:**
- Robust error handling and validation
- Memory-efficient loading for large datasets (chunking)
- Automatic data type inference with manual override options
- Logging of data loading operations and issues
- Data source documentation and metadata preservation

**Example:**
```python
from src.data_loading import load_data_with_validation

# Load CSV with automatic validation
df = load_data_with_validation(
    filepath='data/raw/sales_data.csv',
    expected_columns=['date', 'product_id', 'quantity', 'revenue'],
    parse_dates=['date'],
    validate_schema=True
)
```

### 2. Data Cleaning

**Purpose**: Identify and resolve data quality issues to ensure reliable analysis and modeling.

**Implemented Tasks:**

**Missing Value Handling:**
- Detection and visualization of missingness patterns
- Imputation strategies:
  - Simple: mean, median, mode, constant
  - Advanced: KNN imputation, iterative imputation (MICE)
  - Domain-specific: forward/backward fill for time series
- Deletion strategies: listwise, pairwise, threshold-based

**Outlier Detection and Treatment:**
- Statistical methods: Z-score, IQR, modified Z-score
- Visualization: box plots, scatter plots, distribution plots
- Treatment options: capping (winsorization), transformation, removal, flagging

**Data Type Conversion:**
- Automatic type inference and correction
- Date/time parsing and standardization
- Categorical encoding preparation
- Numeric conversion with error handling

**Duplicate Removal:**
- Identification of exact and fuzzy duplicates
- Configurable duplicate criteria (subset of columns, tolerance)
- Preservation of most recent/complete records

**Data Validation:**
- Range checks (values within expected bounds)
- Consistency checks (cross-field validation)
- Format validation (email, phone, postal codes)
- Referential integrity checks

**Key Features:**
- Comprehensive data quality reports
- Before/after cleaning comparisons
- Audit trails of all cleaning operations
- Configurable cleaning pipelines
- Data profiling and quality metrics

**Example:**
```python
from src.data_cleaning import DataCleaner

# Initialize cleaner with strategies
cleaner = DataCleaner(
    missing_strategy='iterative',
    outlier_method='iqr',
    outlier_action='cap'
)

# Clean data with logging
df_clean = cleaner.fit_transform(df)
cleaner.generate_cleaning_report(save_path='reports/cleaning_report.html')
```

### 3. Data Preparation

**Purpose**: Transform cleaned data into formats suitable for analysis and modeling.

**Implemented Tasks:**

**Feature Engineering:**
- Creation of derived features (ratios, aggregations, differences)
- Temporal features: day of week, month, season, holidays, time since event
- Text features: length, word count, sentiment scores
- Domain-specific transformations
- Polynomial and interaction features
- Binning and discretization

**Encoding Categorical Variables:**
- **Label Encoding**: Ordinal categories with natural ordering
- **One-Hot Encoding**: Nominal categories without ordering
- **Target Encoding**: Mean encoding for high-cardinality features
- **Binary Encoding**: Memory-efficient for many categories
- **Frequency Encoding**: Based on category occurrence
- **Ordinal Encoding**: Custom ordering specification

**Feature Scaling and Normalization:**
- **Standardization (Z-score)**: Mean=0, Std=1; for algorithms assuming normal distribution
- **Min-Max Scaling**: Scale to [0,1] range; preserves relationships
- **Robust Scaling**: Uses median and IQR; robust to outliers
- **Log Transformation**: For right-skewed distributions
- **Box-Cox/Yeo-Johnson**: Power transformations for normality

**Feature Selection:**
- Filter methods: correlation, mutual information, chi-square
- Wrapper methods: recursive feature elimination (RFE)
- Embedded methods: L1 regularization, tree-based importance
- Variance threshold for low-variance features
- Multicollinearity detection (VIF)

**Data Splitting:**
- Train-test-validation splits with stratification
- Time-based splits for temporal data
- Cross-validation fold generation

**Key Features:**
- Pipeline-based transformations for reproducibility
- Fit on training data, transform on test data (no data leakage)
- Reversible transformations where applicable
- Feature importance tracking
- Automated feature documentation

**Example:**
```python
from src.data_preparation import FeatureEngineer, EncodingPipeline

# Create engineered features
engineer = FeatureEngineer()
df_engineered = engineer.create_temporal_features(df, date_column='transaction_date')
df_engineered = engineer.create_interaction_features(df_engineered, ['age', 'income'])

# Encode categorical variables
encoder = EncodingPipeline()
df_encoded = encoder.fit_transform(
    df_engineered,
    onehot_cols=['gender', 'region'],
    target_encode_cols=['product_category']
)
```

### 4. Exploratory Data Analysis (EDA)

**Purpose**: Understand data characteristics, distributions, relationships, and patterns through statistical analysis and visualization.

#### 4.1 Univariate Analysis

**Purpose**: Examine individual variables in isolation to understand their distributions and characteristics.

**Implemented Analyses:**

**For Numerical Variables:**
- **Descriptive Statistics**: mean, median, mode, std, variance, min, max, quartiles
- **Distribution Analysis**: 
  - Histograms with optimal binning (Sturges, Scott, Freedman-Diaconis)
  - Kernel Density Estimation (KDE) plots
  - Q-Q plots for normality assessment
  - Box plots for spread and outliers
- **Shape Metrics**: skewness, kurtosis
- **Statistical Tests**: Shapiro-Wilk (normality), Kolmogorov-Smirnov

**For Categorical Variables:**
- **Frequency Distributions**: value counts, proportions
- **Bar Charts**: ordered by frequency or alphabetically
- **Pie Charts**: for proportional representation (used sparingly)
- **Mode and Entropy**: central tendency and diversity measures

**Key Insights:**
- Central tendency and dispersion
- Distribution shape and symmetry
- Presence of outliers
- Data range and scale
- Missing value patterns

**Example:**
```python
from src.exploratory import UnivariateAnalyzer

analyzer = UnivariateAnalyzer(df)

# Analyze all numerical columns
numerical_summary = analyzer.analyze_numerical_variables(
    variables=['age', 'income', 'purchase_amount'],
    plot=True,
    save_path='reports/figures/univariate/'
)

# Analyze categorical variables
categorical_summary = analyzer.analyze_categorical_variables(
    variables=['gender', 'product_category', 'region'],
    plot=True
)
```

#### 4.2 Bivariate Analysis

**Purpose**: Examine relationships between pairs of variables to identify associations and dependencies.

**Implemented Analyses:**

**Numerical vs Numerical:**
- **Scatter Plots**: relationship visualization with trend lines
- **Correlation Coefficients**: Pearson (linear), Spearman (monotonic), Kendall's tau
- **Joint Plots**: scatter with marginal distributions
- **Hexbin Plots**: for dense point clouds
- **Regression Plots**: with confidence intervals

**Categorical vs Numerical:**
- **Box Plots**: distribution comparison across categories
- **Violin Plots**: distribution shape across categories
- **Strip/Swarm Plots**: individual data points
- **Statistical Tests**: t-test, ANOVA, Mann-Whitney U, Kruskal-Wallis
- **Group Statistics**: mean, median by category

**Categorical vs Categorical:**
- **Contingency Tables**: cross-tabulation with counts/proportions
- **Stacked/Grouped Bar Charts**: category relationships
- **Heatmaps**: frequency matrices
- **Statistical Tests**: Chi-square test of independence, Cramér's V

**Key Insights:**
- Strength and direction of relationships
- Linear vs non-linear associations
- Group differences and comparisons
- Predictive relationships
- Interaction effects

**Example:**
```python
from src.exploratory import BivariateAnalyzer

analyzer = BivariateAnalyzer(df)

# Analyze relationship between income and purchase amount
analyzer.analyze_numerical_pair(
    x='income',
    y='purchase_amount',
    method='scatter',
    add_regression=True,
    color_by='customer_segment'
)

# Compare purchase amount across regions
analyzer.analyze_categorical_numerical(
    categorical='region',
    numerical='purchase_amount',
    method='violin',
    statistical_test='anova'
)
```

#### 4.3 Multivariate Analysis

**Purpose**: Examine complex relationships among three or more variables simultaneously.

**Implemented Analyses:**

**Correlation Analysis:**
- **Correlation Matrices**: Pearson, Spearman correlation heatmaps
- **Partial Correlations**: controlling for confounding variables
- **Network Graphs**: correlation networks with threshold filtering

**Dimensionality Reduction:**
- **Principal Component Analysis (PCA)**: linear dimensionality reduction
- **t-SNE**: non-linear visualization of high-dimensional data
- **UMAP**: preserves local and global structure
- **Factor Analysis**: identify latent variables

**Clustering Analysis:**
- **K-Means**: partition-based clustering
- **Hierarchical Clustering**: dendrograms and cluster hierarchies
- **DBSCAN**: density-based clustering for arbitrary shapes
- **Cluster Profiling**: characterize identified clusters

**Advanced Visualizations:**
- **Pair Plots**: all pairwise relationships (scatter plot matrices)
- **Parallel Coordinates**: multivariate patterns
- **3D Scatter Plots**: three-variable relationships
- **Bubble Plots**: four variables (x, y, size, color)
- **Facet Grids**: small multiples across categorical dimensions

**Statistical Methods:**
- **Multiple Regression Analysis**: multivariate relationships
- **ANOVA/MANOVA**: multiple group comparisons
- **Variance Inflation Factor (VIF)**: multicollinearity detection

**Key Insights:**
- Complex interaction effects
- Hidden patterns and structures
- Multicollinearity issues
- Data dimensionality understanding
- Natural groupings in data

**Example:**
```python
from src.exploratory import MultivariateAnalyzer

analyzer = MultivariateAnalyzer(df)

# Correlation analysis with heatmap
correlation_matrix = analyzer.correlation_analysis(
    variables=['age', 'income', 'purchase_amount', 'loyalty_score'],
    method='pearson',
    plot_heatmap=True,
    annot=True
)

# PCA for dimensionality reduction
pca_results = analyzer.perform_pca(
    n_components=3,
    plot_variance_explained=True,
    plot_loadings=True
)

# Clustering analysis
cluster_labels = analyzer.kmeans_clustering(
    n_clusters=4,
    features=['age', 'income', 'purchase_frequency'],
    plot_clusters=True,
    profile_clusters=True
)
```

### 5. Data Visualization

**Purpose**: Create compelling visual representations that communicate insights effectively.

**Implemented Visualizations:**

#### Distribution Plots
- **Histograms**: frequency distributions with configurable bins
- **KDE Plots**: smooth probability density estimation
- **Box Plots**: quartile-based distribution with outliers
- **Violin Plots**: distribution shape across categories
- **ECDF Plots**: empirical cumulative distribution functions
- **Rug Plots**: individual data point markers

#### Relationship Plots
- **Scatter Plots**: two-variable relationships with customization
- **Line Plots**: temporal trends and trajectories
- **Regression Plots**: relationships with fitted models
- **Residual Plots**: model diagnostics
- **Hexbin/2D Histograms**: dense scatter data
- **Contour Plots**: continuous bivariate distributions

#### Comparison Plots
- **Bar Charts**: categorical comparisons (grouped, stacked)
- **Grouped Box/Violin Plots**: distribution comparisons
- **Heatmaps**: matrix-based comparisons
- **Radar/Spider Charts**: multivariate profile comparisons
- **Slope Charts**: before/after comparisons

#### Temporal Analysis
- **Time Series Plots**: trends over time with multiple series
- **Seasonal Decomposition**: trend, seasonality, residuals
- **Rolling Statistics**: moving averages and windows
- **Calendar Heatmaps**: temporal patterns by day/month
- **Autocorrelation Plots**: temporal dependencies

#### Geospatial Visualization
- **Choropleth Maps**: regional data with color encoding
- **Scatter Maps**: point-based geographic data
- **Bubble Maps**: size-encoded geographic data
- **Heat Maps**: density of geographic points
- **Flow Maps**: movement and connections

#### Interactive Dashboards
- **Plotly Dashboards**: interactive web-based visualizations
- **Dash Applications**: full-featured analytics dashboards
- **Widgets**: dropdowns, sliders, date pickers for filtering
- **Linked Brushing**: coordinated multi-view interactions

**Visualization Principles Applied:**
- Clear, descriptive titles and labels
- Appropriate color schemes (colorblind-friendly options)
- Consistent styling and themes
- Proper scaling and aspect ratios
- Minimal chart junk, maximum data-ink ratio
- Contextual annotations and references

**Example:**
```python
from src.visualization import VisualizationSuite

viz = VisualizationSuite(style='seaborn-v0_8-darkgrid', palette='husl')

# Create comprehensive distribution analysis
viz.plot_distribution_suite(
    data=df,
    variable='purchase_amount',
    group_by='customer_segment',
    save_path='reports/figures/distributions/'
)

# Time series visualization with trends
viz.plot_time_series(
    data=df,
    date_column='date',
    value_column='daily_revenue',
    add_trend=True,
    add_seasonality=True,
    forecast_periods=30
)

# Interactive dashboard
viz.create_interactive_dashboard(
    data=df,
    metrics=['revenue', 'customers', 'conversion_rate'],
    dimensions=['date', 'region', 'product_category'],
    save_path='reports/dashboard.html'
)
```

### 6. Advanced Data Science Activities

**Additional Tasks Implemented:**

#### Statistical Testing
- Hypothesis testing (t-tests, ANOVA, chi-square)
- A/B testing framework with power analysis
- Multiple comparison corrections (Bonferroni, FDR)
- Non-parametric tests (Mann-Whitney, Kruskal-Wallis)

#### Time Series Analysis
- Stationarity testing (ADF, KPSS)
- Autocorrelation and partial autocorrelation
- Seasonal decomposition (additive, multiplicative)
- Trend detection and change point analysis

#### Feature Importance
- Permutation importance
- SHAP values for model interpretation
- Partial dependence plots
- Tree-based feature importance

#### Data Quality Monitoring
- Automated data quality reports
- Drift detection for production data
- Anomaly detection in data distributions
- Data lineage tracking

## Getting Started

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook/Lab
pip or conda for package management
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/data-science-workflow.git
cd data-science-workflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate data-science-env
```

### Quick Start
```python
# Complete workflow example
from src.data_loading import DataLoader
from src.data_cleaning import DataCleaner
from src.exploratory import ComprehensiveEDA
from src.visualization import ReportGenerator

# 1. Load data
loader = DataLoader()
df = loader.load_csv('data/raw/dataset.csv', parse_dates=['date'])

# 2. Clean data
cleaner = DataCleaner()
df_clean = cleaner.fit_transform(df)

# 3. Perform EDA
eda = ComprehensiveEDA(df_clean)
eda.generate_full_report(save_path='reports/eda_report.html')

# 4. Create visualizations
report = ReportGenerator(df_clean)
report.create_executive_summary(
    target_variable='revenue',
    save_path='reports/executive_summary.pdf'
)
```

### Project Workflow

**Step 1: Data Loading and Initial Inspection**
```bash
jupyter notebook notebooks/01_data_loading/loading_csv_files.ipynb
```
- Load data from various sources
- Perform initial data inspection
- Document data sources and metadata

**Step 2: Data Cleaning**
```bash
jupyter notebook notebooks/02_data_cleaning/handling_missing_values.ipynb
```
- Handle missing values systematically
- Detect and treat outliers
- Remove duplicates and fix data types

**Step 3: Data Preparation**
```bash
jupyter notebook notebooks/03_data_preparation/feature_engineering.ipynb
```
- Engineer relevant features
- Encode categorical variables
- Scale and normalize features

**Step 4: Exploratory Analysis**
```bash
jupyter notebook notebooks/04_exploratory_analysis/univariate_analysis.ipynb
```
- Conduct univariate, bivariate, and multivariate analyses
- Perform statistical tests
- Document key findings

**Step 5: Visualization**
```bash
jupyter notebook notebooks/05_visualization/distribution_plots.ipynb
```
- Create comprehensive visualizations
- Build interactive dashboards
- Generate presentation-ready charts

## Key Features

**Comprehensive Coverage**: Complete data science workflow from raw data to insights, covering all essential steps with professional implementations.

**Modular Design**: Reusable functions and classes organized by workflow stage, enabling easy integration into custom projects.

**Best Practices**: Industry-standard approaches to data handling, analysis, and visualization with emphasis on reproducibility and code quality.

**Educational Focus**: Well-documented code with explanations of methods, assumptions, and interpretations suitable for learning and teaching.

**Production-Ready**: Includes logging, error handling, input validation, and configuration management for real-world applications.

**Extensive Documentation**: Each notebook contains markdown cells explaining concepts, methodologies, and interpretations of results.

**Automated Reporting**: Generate HTML and PDF reports summarizing analyses, complete with statistics, visualizations, and insights.

## Datasets Included

The repository includes sample datasets across various domains:

1. **Sales Data**: transactional data with temporal patterns
2. **Customer Data**: demographic and behavioral information
3. **Financial Data**: stock prices, economic indicators
4. **Healthcare Data**: patient records (anonymized)
5. **Social Media Data**: engagement metrics, sentiment scores
6. **IoT Sensor Data**: time-series measurements

Each dataset includes a README with:
- Data dictionary (column descriptions, types, ranges)
- Source and collection methodology
- Known issues and limitations
- Recommended analyses

## Best Practices Demonstrated

**Data Handling:**
- Preserve raw data; always work on copies
- Document all transformations with clear rationale
- Version datasets when making significant changes
- Maintain data lineage and audit trails

**Analysis:**
- State assumptions explicitly
- Use appropriate statistical tests for data types
- Check assumptions (normality, independence, etc.)
- Report effect sizes alongside p-values
- Consider practical significance, not just statistical

**Visualization:**
- Choose appropriate chart types for data and message
- Use consistent color schemes and styling
- Label axes clearly with units
- Provide context through titles and annotations
- Make visualizations accessible (colorblind-friendly)

**Code Quality:**
- Modular, reusable functions
- Comprehensive docstrings and comments
- Type hints for function signatures
- Unit tests for critical functions
- Version control with meaningful commits

**Reproducibility:**
- Set random seeds for stochastic processes
- Document environment and dependencies
- Use relative paths, not absolute
- Create requirements.txt or environment.yml
- Include data generation scripts where applicable

## Common Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| **Large datasets don't fit in memory** | Use chunking, Dask for parallel processing, or sample for exploration |
| **High dimensionality** | Apply dimensionality reduction (PCA, feature selection) before analysis |
| **Imbalanced classes** | Use stratified sampling, SMOTE, or class weights in modeling |
| **Missing data patterns** | Analyze missingness mechanism (MCAR, MAR, MNAR) before imputation |
| **Multiple hypothesis testing** | Apply correction methods (Bonferroni, FDR) to control family-wise error |
| **Non-normal distributions** | Use transformations (log, Box-Cox) or non-parametric methods |
| **Outliers influencing results** | Use robust statistics (median, IQR) or outlier-resistant methods |
| **Correlation vs causation** | Design proper experiments or use causal inference methods |

## Performance Optimization

**For Large Datasets:**
- Use categorical dtype for string columns (memory reduction)
- Read only necessary columns with `usecols` parameter
- Use chunking: `pd.read_csv(chunksize=10000)`
- Leverage Dask or PySpark for distributed computing
- Convert to Parquet format for faster I/O

**For Analysis:**
- Vectorize operations instead of loops
- Use NumPy for numerical computations
- Profile code to identify bottlenecks
- Cache expensive computations
- Parallelize independent operations

## Visualization Gallery

The repository includes examples of:
- 50+ different plot types
- Custom themes and color palettes
- Publication-quality figures
- Interactive web dashboards
- Animated visualizations for presentations
- Infographic-style summary visualizations

View the complete gallery at `reports/figures/gallery/`

## Dependencies

Core libraries:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Statistical functions and tests
- **scikit-learn**: Machine learning utilities and preprocessing
- **matplotlib**: Static plotting
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations
- **dash**: Interactive dashboards
- **statsmodels**: Statistical modeling and tests
- **jupyter**: Interactive notebooks

See `requirements.txt` for complete dependency list with versions.

## Contributing

Contributions are welcome! Areas for enhancement:

- Additional datasets from diverse domains
- More advanced statistical methods
- Industry-specific analysis templates
- Additional visualization types
- Performance optimization techniques
- Cloud-based data loading examples
- Real-time data streaming examples
- Automated feature engineering tools

Please open an issue to discuss before submitting pull requests.

## Project Organization

This repository follows a standardized data science project structure inspired by Cookiecutter Data Science, promoting reproducibility and collaboration.

## Learning Path

**Beginners**: Start with notebooks  (loading, cleaning, preparation)
**Intermediate**: Focus on notebooks  (exploratory analysis) and 05 (visualization)
**Advanced**: Explore notebooks (advanced tasks) and extend with custom analyses

Each notebook is self-contained with explanations suitable for different skill levels.

## Resources

**Further Reading:**
- *Python for Data Analysis* by Wes McKinney
- *Storytelling with Data* by Cole Nussbaumer Knaflic
- *The Visual Display of Quantitative Information* by Edward Tufte
- pandas documentation: https://pandas.pydata.org
- seaborn gallery: https://seaborn.pydata.org/examples/

## Acknowledgments

This repository synthesizes best practices from the data science community, academic research, and industry applications. It aims to provide a comprehensive, practical reference for executing professional data science workflows.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out through the repository's discussion forum.

email address: bahmadnazif@gmail.com                  LINKEDLN: Muhammad Nazif Ahmad

**Note**: This repository focuses on structured tabular data. For specialized domains like computer vision, natural language processing, or deep learning, additional domain-specific preprocessing and analysis techniques would be required.
