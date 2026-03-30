"""
Space Mission Financial Planning - Enhanced Agentic AI System
With CSV Data Ingestion, Distribution Learning, and Visualization
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline

# Statistical imports
from scipy import stats
from scipy.stats import norm, lognorm, gamma, expon, weibull_min
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# For Jupyter compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    print("Note: Install nest_asyncio for better Jupyter compatibility: pip install nest_asyncio")

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class MissionRequirements:
    """Core mission requirements and specifications"""
    mission_id: str
    mission_type: str  # LEO, GEO, Deep Space, Lunar, Mars
    duration_months: int
    payload_mass_kg: float
    launch_date: datetime
    technology_readiness_level: int
    crew_size: int
    science_objectives: List[str]
    constraints: Dict[str, Any]
    
    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> 'MissionRequirements':
        """Create MissionRequirements from a DataFrame row"""
        return cls(
            mission_id=str(row.get('mission_id', f"MISSION-{datetime.now().strftime('%Y%m%d')}")),
            mission_type=str(row.get('mission_type', 'LEO')),
            duration_months=int(row.get('duration_months', 12)),
            payload_mass_kg=float(row.get('payload_mass_kg', 1000)),
            launch_date=pd.to_datetime(row.get('launch_date', datetime.now())),
            technology_readiness_level=int(row.get('trl', 7)),
            crew_size=int(row.get('crew_size', 0)),
            science_objectives=str(row.get('objectives', '')).split(';') if pd.notna(row.get('objectives')) else [],
            constraints=json.loads(row.get('constraints', '{}')) if isinstance(row.get('constraints'), str) else {}
        )

@dataclass
class DistributionParameters:
    """Parameters for statistical distributions"""
    distribution_type: str
    parameters: Dict[str, float]
    goodness_of_fit: float
    sample_size: int

@dataclass
class TrainingDataset:
    """Container for training data and learned distributions"""
    raw_data: pd.DataFrame
    feature_distributions: Dict[str, DistributionParameters]
    cost_model: Any
    risk_model: Any
    optimization_model: Any
    metadata: Dict[str, Any]

# ============================================================================
# DATA INGESTION AND DISTRIBUTION LEARNING
# ============================================================================

class DataIngestionPipeline:
    """Pipeline for ingesting CSV data and learning distributions"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_distributions = {}
        self.models = {}
        self.feature_columns = []
        
    def load_csv_data(self, filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """Load and preprocess CSV data"""
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"✓ Loaded {len(df)} records from {filepath}")
            print(f"  Columns: {', '.join(df.columns.tolist())}")
            # Apply schema mapping for messy spreadsheets
            df = self.schema_mapping_agent(df)
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            # Generate synthetic data if file not found
            return self.generate_synthetic_data()
    
    def schema_mapping_agent(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use AI-like logic to map messy column names to expected MissionRequirements fields"""
        import difflib
        
        # Expected fields and their possible variations
        field_mappings = {
            'mission_id': ['mission_id', 'id', 'mission', 'name', 'mission_name', 'project_id'],
            'mission_type': ['mission_type', 'type', 'mission', 'category', 'orbit_type'],
            'duration_months': ['duration_months', 'duration', 'months', 'time', 'period', 'length'],
            'payload_mass_kg': ['payload_mass_kg', 'payload', 'mass', 'weight', 'kg', 'payload_mass'],
            'launch_date': ['launch_date', 'date', 'launch', 'start_date', 'begin_date'],
            'technology_readiness_level': ['trl', 'technology_readiness_level', 'readiness', 'tech_level', 'maturity'],
            'crew_size': ['crew_size', 'crew', 'people', 'personnel', 'astronauts'],
            'science_objectives': ['science_objectives', 'objectives', 'goals', 'science', 'missions'],
            'constraints': ['constraints', 'limits', 'requirements', 'restrictions']
        }
        
        column_mapping = {}
        used_columns = set()
        
        for expected_field, keywords in field_mappings.items():
            best_match = None
            best_score = 0
            
            for col in df.columns:
                if col in used_columns:
                    continue
                col_lower = col.lower().replace('_', ' ').replace('-', ' ')
                
                # Check for exact matches first
                if expected_field.lower() in col_lower:
                    best_match = col
                    best_score = 1.0
                    break
                
                # Check keyword matches
                for keyword in keywords:
                    if keyword in col_lower:
                        score = len(keyword) / len(col_lower)  # Simple score
                        if score > best_score:
                            best_score = score
                            best_match = col
                            break
                
                # Use difflib for fuzzy matching
                if not best_match:
                    ratios = [difflib.SequenceMatcher(None, expected_field.lower(), col_lower).ratio()]
                    max_ratio = max(ratios)
                    if max_ratio > 0.6 and max_ratio > best_score:
                        best_score = max_ratio
                        best_match = col
            
            if best_match:
                column_mapping[expected_field] = best_match
                used_columns.add(best_match)
                print(f"✓ Mapped '{best_match}' -> '{expected_field}'")
            else:
                print(f"⚠ No mapping found for '{expected_field}', will use default")
        
        # Rename columns
        df_renamed = df.rename(columns=column_mapping)
        
        # Ensure all expected columns exist, add defaults if missing
        for field in field_mappings.keys():
            if field not in df_renamed.columns:
                if field in ['science_objectives', 'constraints']:
                    df_renamed[field] = '[]' if field == 'science_objectives' else '{}'
                elif field == 'launch_date':
                    df_renamed[field] = datetime.now()
                else:
                    df_renamed[field] = None
        
        return df_renamed
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic mission data for demonstration"""
        np.random.seed(42)
        
        mission_types = ['LEO', 'GEO', 'Lunar', 'Mars', 'Deep Space']
        
        data = {
            'mission_id': [f'MISSION-{i:04d}' for i in range(n_samples)],
            'mission_type': np.random.choice(mission_types, n_samples),
            'duration_months': np.random.lognormal(3.0, 0.5, n_samples).clip(6, 120),
            'payload_mass_kg': np.random.lognormal(7.0, 1.0, n_samples).clip(100, 50000),
            'trl': np.random.choice(range(4, 10), n_samples, p=[0.05, 0.1, 0.2, 0.3, 0.25, 0.1]),
            'crew_size': np.random.choice([0, 2, 4, 6], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
            'total_cost_millions': None,  # Will be calculated
            'cost_overrun_percent': np.random.gamma(2, 5, n_samples).clip(0, 100),
            'schedule_delay_months': np.random.exponential(3, n_samples).clip(0, 24),
            'mission_success': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),
            'risk_score': np.random.beta(2, 5, n_samples) * 100,
            'launch_year': np.random.choice(range(2010, 2025), n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate total cost based on features (synthetic relationship)
        base_cost = 100  # Base cost in millions
        df['total_cost_millions'] = (
            base_cost +
            df['duration_months'] * 2.5 +
            np.log1p(df['payload_mass_kg']) * 10 +
            (10 - df['trl']) * 15 +
            df['crew_size'] * 50 +
            np.random.normal(0, 20, n_samples)
        ).clip(50, 2000)
        
        # Add some categorical features
        df['contractor'] = np.random.choice(['SpaceX', 'Boeing', 'Lockheed', 'Northrop', 'Blue Origin'], n_samples)
        df['launch_site'] = np.random.choice(['Kennedy', 'Vandenberg', 'Baikonur', 'Kourou'], n_samples)
        
        print(f"✓ Generated synthetic dataset with {n_samples} samples")
        return df
    
    def analyze_distributions(self, df: pd.DataFrame, numeric_columns: List[str] = None) -> Dict[str, DistributionParameters]:
        """Analyze and fit distributions to numeric columns"""
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        distributions_to_test = [
            ('normal', norm),
            ('lognormal', lognorm),
            ('gamma', gamma),
            ('exponential', expon),
            ('weibull', weibull_min)
        ]
        
        results = {}
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
                
            data = df[col].dropna()
            if len(data) < 10:
                continue
            
            best_dist = None
            best_params = None
            best_ks_statistic = float('inf')
            
            for dist_name, dist_func in distributions_to_test:
                try:
                    # Fit distribution
                    params = dist_func.fit(data)
                    
                    # Kolmogorov-Smirnov test
                    ks_statistic, p_value = stats.kstest(data, lambda x: dist_func.cdf(x, *params))
                    
                    if ks_statistic < best_ks_statistic:
                        best_ks_statistic = ks_statistic
                        best_dist = dist_name
                        best_params = params
                
                except Exception:
                    continue
            
            if best_dist:
                results[col] = DistributionParameters(
                    distribution_type=best_dist,
                    parameters={'params': best_params},
                    goodness_of_fit=1 - best_ks_statistic,  # Convert to goodness score
                    sample_size=len(data)
                )
                
                print(f"  {col}: {best_dist} distribution (fit score: {results[col].goodness_of_fit:.3f})")
        
        return results
    
    def train_cost_model(self, df: pd.DataFrame) -> RandomForestRegressor:
        """Train a cost estimation model"""
        
        # Prepare features
        feature_cols = ['duration_months', 'payload_mass_kg', 'trl', 'crew_size']
        
        # Check if we have the necessary columns
        available_features = [col for col in feature_cols if col in df.columns]
        if 'total_cost_millions' not in df.columns or len(available_features) == 0:
            print("Warning: Insufficient data for training cost model")
            return None
        
        # One-hot encode categorical variables if present
        categorical_cols = ['mission_type', 'contractor', 'launch_site']
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                available_features.extend(dummies.columns.tolist())
        
        X = df[available_features].fillna(0)
        y = df['total_cost_millions'].fillna(df['total_cost_millions'].mean())
        
        # Save feature columns for prediction alignment
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✓ Cost Model Performance:")
        print(f"  MAE: ${mae:.2f}M")
        print(f"  R²: {r2:.3f}")
        
        return model
    
    def train_risk_model(self, df: pd.DataFrame) -> GradientBoostingRegressor:
        """Train a risk assessment model"""
        
        feature_cols = ['duration_months', 'payload_mass_kg', 'trl', 'crew_size', 'total_cost_millions']
        available_features = [col for col in feature_cols if col in df.columns]
        
        if 'risk_score' not in df.columns or len(available_features) < 2:
            print("Warning: Insufficient data for training risk model")
            return None
        
        X = df[available_features].fillna(0)
        y = df['risk_score'].fillna(df['risk_score'].mean())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✓ Risk Model Performance:")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.3f}")
        
        return model

# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class VisualizationEngine:
    """Generate various charts and visualizations"""
    
    def __init__(self):
        self.color_scheme = px.colors.qualitative.Set3
        
    def create_cost_breakdown_chart(self, wbs_costs: Dict[str, float], title: str = "Mission Cost Breakdown") -> go.Figure:
        """Create an interactive pie chart for cost breakdown"""
        
        labels = list(wbs_costs.keys())
        values = list(wbs_costs.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=self.color_scheme),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Cost: $%{value:,.2f}M<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=500,
            template='plotly_dark'
        )
        
        return fig
    
    def create_risk_matrix(self, risks: List[Dict]) -> go.Figure:
        """Create a risk probability-impact matrix"""
        
        if not risks:
            return self._empty_figure("No risks identified")
        
        probabilities = [r.get('probability', 0) * 100 for r in risks]
        impacts = [r.get('impact', 0) / 1_000_000 for r in risks]  # Convert to millions
        labels = [r.get('description', f"Risk {i}") for i, r in enumerate(risks)]
        categories = [r.get('category', 'Unknown') for r in risks]
        
        # Create color map for categories
        unique_categories = list(set(categories))
        color_map = {cat: self.color_scheme[i % len(self.color_scheme)] 
                    for i, cat in enumerate(unique_categories)}
        colors = [color_map[cat] for cat in categories]
        
        fig = go.Figure()
        
        for cat in unique_categories:
            cat_mask = [c == cat for c in categories]
            cat_probs = [p for p, m in zip(probabilities, cat_mask) if m]
            cat_impacts = [i for i, m in zip(impacts, cat_mask) if m]
            cat_labels = [l for l, m in zip(labels, cat_mask) if m]
            
            fig.add_trace(go.Scatter(
                x=cat_probs,
                y=cat_impacts,
                mode='markers+text',
                name=cat,
                text=[l[:20] + '...' if len(l) > 20 else l for l in cat_labels],
                textposition='top center',
                marker=dict(size=15, color=color_map[cat]),
                hovertemplate='<b>%{text}</b><br>Probability: %{x:.1f}%<br>Impact: $%{y:.1f}M<extra></extra>'
            ))
        
        # Add risk zones
        fig.add_shape(type="rect", x0=0, y0=0, x1=33, y1=33,
                     fillcolor="green", opacity=0.2, layer="below")
        fig.add_shape(type="rect", x0=33, y0=0, x1=66, y1=66,
                     fillcolor="yellow", opacity=0.2, layer="below")
        fig.add_shape(type="rect", x0=66, y0=0, x1=100, y1=100,
                     fillcolor="red", opacity=0.2, layer="below")
        
        fig.update_layout(
            title="Risk Probability-Impact Matrix",
            xaxis_title="Probability (%)",
            yaxis_title="Impact ($M)",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, max(impacts) * 1.2] if impacts else [0, 100]),
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_optimization_comparison(self, original: Dict[str, float], 
                                     optimized: Dict[str, float]) -> go.Figure:
        """Create a comparison chart between original and optimized budgets"""
        
        categories = list(original.keys())
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Original Budget',
            x=categories,
            y=list(original.values()),
            marker_color='lightblue',
            hovertemplate='<b>%{x}</b><br>Original: $%{y:,.2f}M<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Optimized Budget',
            x=categories,
            y=list(optimized.values()),
            marker_color='darkblue',
            hovertemplate='<b>%{x}</b><br>Optimized: $%{y:,.2f}M<extra></extra>'
        ))
        
        fig.update_layout(
            title="Budget Optimization Comparison",
            xaxis_title="Category",
            yaxis_title="Budget ($M)",
            barmode='group',
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_confidence_interval_chart(self, estimates: Dict[str, Tuple[float, float, float]]) -> go.Figure:
        """Create a chart showing estimates with confidence intervals"""
        
        categories = list(estimates.keys())
        lower_bounds = [v[0] / 1_000_000 for v in estimates.values()]
        estimates_val = [v[1] / 1_000_000 for v in estimates.values()]
        upper_bounds = [v[2] / 1_000_000 for v in estimates.values()]
        
        fig = go.Figure()
        
        # Add the main estimate line
        fig.add_trace(go.Scatter(
            x=categories,
            y=estimates_val,
            mode='lines+markers',
            name='Estimate',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Estimate: $%{y:,.2f}M<extra></extra>'
        ))
        
        # Add confidence interval as filled area
        fig.add_trace(go.Scatter(
            x=categories + categories[::-1],
            y=upper_bounds + lower_bounds[::-1],
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title="Cost Estimates with 90% Confidence Intervals",
            xaxis_title="Phase",
            yaxis_title="Cost ($M)",
            template='plotly_dark',
            height=500
        )
        
        return fig
    
    def create_distribution_plots(self, df: pd.DataFrame, distributions: Dict[str, DistributionParameters]) -> go.Figure:
        """Create distribution plots for key parameters"""
        
        n_plots = min(6, len(distributions))
        if n_plots == 0:
            return self._empty_figure("No distributions to plot")
        
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(distributions.keys())[:n_plots]
        )
        
        for idx, (col_name, dist_params) in enumerate(list(distributions.items())[:n_plots]):
            row = idx // cols + 1
            col = idx % cols + 1
            
            data = df[col_name].dropna()
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=data,
                    nbinsx=30,
                    name=f'{col_name} (actual)',
                    histnorm='probability density',
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add fitted distribution
            x_range = np.linspace(data.min(), data.max(), 100)
            
            if dist_params.distribution_type == 'normal':
                y_fit = norm.pdf(x_range, *dist_params.parameters['params'])
            elif dist_params.distribution_type == 'lognormal':
                y_fit = lognorm.pdf(x_range, *dist_params.parameters['params'])
            elif dist_params.distribution_type == 'gamma':
                y_fit = gamma.pdf(x_range, *dist_params.parameters['params'])
            elif dist_params.distribution_type == 'exponential':
                y_fit = expon.pdf(x_range, *dist_params.parameters['params'])
            elif dist_params.distribution_type == 'weibull':
                y_fit = weibull_min.pdf(x_range, *dist_params.parameters['params'])
            else:
                continue
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_fit,
                    mode='lines',
                    name=f'{dist_params.distribution_type} fit',
                    line=dict(color='red', width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Parameter Distributions with Fitted Curves",
            template='plotly_dark',
            height=200 * rows,
            showlegend=False
        )
        
        return fig
    
    def create_timeline_gantt(self, phases: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create a Gantt chart for mission timeline"""
        
        tasks = []
        for phase_name, phase_info in phases.items():
            tasks.append(dict(
                Task=phase_name,
                Start=phase_info.get('start', datetime.now()),
                Finish=phase_info.get('end', datetime.now() + timedelta(days=30)),
                Resource=phase_info.get('resource', 'Team A')
            ))
        
        df_gantt = pd.DataFrame(tasks)
        
        fig = px.timeline(
            df_gantt, 
            x_start="Start", 
            x_end="Finish", 
            y="Task",
            color="Resource",
            title="Mission Timeline",
            template='plotly_dark'
        )
        
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=400)
        
        return fig
    
    def create_dashboard(self, results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create a comprehensive dashboard with multiple visualizations"""
        
        dashboard = {}
        
        # Cost breakdown
        if 'cost_analysis' in results and 'wbs_breakdown' in results['cost_analysis']:
            dashboard['cost_breakdown'] = self.create_cost_breakdown_chart(
                results['cost_analysis']['wbs_breakdown']
            )
        
        # Risk matrix
        if 'risk_analysis' in results and 'identified_risks' in results['risk_analysis']:
            dashboard['risk_matrix'] = self.create_risk_matrix(
                results['risk_analysis']['identified_risks']
            )
        
        # Budget optimization
        if 'optimization' in results and 'cost_analysis' in results:
            if 'optimized_allocation' in results['optimization'] and 'wbs_breakdown' in results['cost_analysis']:
                dashboard['optimization'] = self.create_optimization_comparison(
                    results['cost_analysis']['wbs_breakdown'],
                    results['optimization']['optimized_allocation']
                )
        
        # Summary metrics
        dashboard['summary'] = self._create_summary_metrics(results)
        
        return dashboard
    
    def _create_summary_metrics(self, results: Dict[str, Any]) -> go.Figure:
        """Create a summary metrics visualization"""
        
        fig = go.Figure()
        
        # Extract key metrics
        metrics = {
            'Total Cost': results.get('cost_analysis', {}).get('total_estimated_cost', 0) / 1_000_000,
            'Risk Score': results.get('risk_analysis', {}).get('risk_score', 0),
            'Savings': results.get('optimization', {}).get('expected_savings', 0) / 1_000_000,
            'Confidence': results.get('confidence_score', 0) * 100
        }
        
        # Create gauge charts
        for i, (label, value) in enumerate(metrics.items()):
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': label},
                domain={'x': [i*0.25, (i+1)*0.25], 'y': [0, 1]},
                gauge={'axis': {'range': [None, value * 1.5] if label != 'Risk Score' else [0, 100]}}
            ))
        
        fig.update_layout(
            title="Executive Summary Metrics",
            template='plotly_dark',
            height=300
        )
        
        return fig
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            template='plotly_dark',
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
    
    def save_all_figures(self, dashboard: Dict[str, go.Figure], output_dir: str = "./outputs"):
        """Save all figures to high-resolution PNG files for presentations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in dashboard.items():
            # Save as high-resolution PNG
            try:
                fig.write_image(f"{output_dir}/{name}.png", width=1920, height=1080, scale=2)
                print(f"✓ Saved {name}.png (1920x1080, 2x scale)")
            except ImportError:
                print(f"⚠ Install kaleido for PNG export: pip install kaleido")
                # Fallback to HTML
                fig.write_html(f"{output_dir}/{name}.html")
            except Exception as e:
                print(f"Error saving {name}: {e}")
        
        print(f"✓ Saved {len(dashboard)} visualizations to {output_dir}/")

# ============================================================================
# ENHANCED AGENTS WITH ML MODELS
# ============================================================================

class EnhancedCostEstimationAgent:
    """Enhanced cost estimation agent using trained ML models"""
    
    def __init__(self, trained_model: Any = None, distributions: Dict = None, feature_columns: List[str] = None):
        self.ml_model = trained_model
        self.distributions = distributions or {}
        self.feature_columns = feature_columns or []
        self.historical_costs = []
    
    def estimate_with_ml(self, mission_req: Dict) -> Dict[str, Any]:
        """Use ML model for cost estimation"""
        
        if self.ml_model is None:
            # Fallback to parametric estimation
            return self._parametric_estimation(mission_req)
        
        try:
            # Prepare features for ML model
            features = self._prepare_features(mission_req)
            
            # Get point estimate
            point_estimate = self.ml_model.predict([features])[0] * 1_000_000
            
            # Generate confidence interval using bootstrap
            bootstrap_estimates = []
            for _ in range(100):
                # Sample from distributions
                sampled_features = self._sample_from_distributions(features)
                estimate = self.ml_model.predict([sampled_features])[0] * 1_000_000
                bootstrap_estimates.append(estimate)
            
            lower_bound = np.percentile(bootstrap_estimates, 5)
            upper_bound = np.percentile(bootstrap_estimates, 95)
            
            return {
                'point_estimate': point_estimate,
                'confidence_interval': (lower_bound, upper_bound),
                'confidence_level': 0.90,
                'method': 'ML-based estimation'
            }
        
        except Exception as e:
            print(f"ML estimation failed: {e}, falling back to parametric")
            return self._parametric_estimation(mission_req)
    
    def _prepare_features(self, mission_req: Dict) -> List[float]:
        """Match training feature structure exactly"""
        
        # Create base row
        row = {
            'duration_months': mission_req.get('duration_months', 12),
            'payload_mass_kg': mission_req.get('payload_mass_kg', 1000),
            'trl': mission_req.get('technology_readiness_level', 7),
            'crew_size': mission_req.get('crew_size', 0),
            'mission_type': mission_req.get('mission_type', 'LEO')
        }
        
        df = pd.DataFrame([row])
        
        # One-hot encode like training
        df = pd.get_dummies(df)
        
        # Align with training columns
        for col in self.feature_columns:
            if col not in df:
                df[col] = 0
        
        df = df[self.feature_columns]
        
        return df.iloc[0].values
    
    def _sample_from_distributions(self, features: List[float]) -> List[float]:
        """Sample from learned distributions for uncertainty quantification"""
        sampled = []
        for i, val in enumerate(features):
            if f'feature_{i}' in self.distributions:
                dist_params = self.distributions[f'feature_{i}']
                if dist_params.distribution_type == 'normal':
                    sampled_val = np.random.normal(*dist_params.parameters['params'])
                else:
                    sampled_val = val + np.random.normal(0, val * 0.1)  # 10% noise
                sampled.append(sampled_val)
            else:
                sampled.append(val)
        return sampled
    
    def _parametric_estimation(self, mission_req: Dict) -> Dict[str, Any]:
        """Fallback parametric estimation"""
        base_rate = 100_000_000
        mass_factor = mission_req.get('payload_mass_kg', 1000) * 50_000
        duration_factor = mission_req.get('duration_months', 12) * 5_000_000
        trl_factor = (10 - mission_req.get('technology_readiness_level', 7)) * 10_000_000
        
        total = base_rate + mass_factor + duration_factor + trl_factor
        
        return {
            'point_estimate': total,
            'confidence_interval': (total * 0.85, total * 1.15),
            'confidence_level': 0.90,
            'method': 'Parametric estimation'
        }

# ============================================================================
# INTEGRATED ORCHESTRATOR WITH DATA LEARNING
# ============================================================================

class EnhancedAgentOrchestrator:
    """Enhanced orchestrator with data ingestion and visualization"""
    
    def __init__(self, training_data_path: str = None):
        self.data_pipeline = DataIngestionPipeline()
        self.viz_engine = VisualizationEngine()
        self.training_data = None
        self.trained_models = {}
        
        if training_data_path:
            self.load_and_train(training_data_path)
    
    def load_and_train(self, csv_path: str):
        """Load CSV data and train all models"""
        
        print("\n" + "="*60)
        print("DATA INGESTION AND MODEL TRAINING")
        print("="*60)
        
        # Load data
        df = self.data_pipeline.load_csv_data(csv_path) if csv_path else self.data_pipeline.generate_synthetic_data()
        
        # Analyze distributions
        print("\n📊 Analyzing parameter distributions...")
        distributions = self.data_pipeline.analyze_distributions(df)
        
        # Train models
        print("\n🤖 Training ML models...")
        cost_model = self.data_pipeline.train_cost_model(df)
        risk_model = self.data_pipeline.train_risk_model(df)
        
        # Store results
        self.training_data = df
        self.trained_models = {
            'cost_model': cost_model,
            'risk_model': risk_model,
            'distributions': distributions
        }
        
        # Create distribution visualizations
        print("\n📈 Creating distribution visualizations...")
        dist_fig = self.viz_engine.create_distribution_plots(df, distributions)
        
        return dist_fig
    
    def process_mission_with_visualization(self, mission_req: MissionRequirements) -> Tuple[Dict, Dict[str, go.Figure]]:
        """Process mission and generate visualizations"""
        
        print("\n" + "="*60)
        print("ENHANCED MISSION ANALYSIS WITH ML")
        print("="*60)
        
        # Use enhanced agent if model is trained
        if self.trained_models.get('cost_model'):
            cost_agent = EnhancedCostEstimationAgent(
                self.trained_models['cost_model'],
                self.trained_models['distributions'],
                self.data_pipeline.feature_columns
            )
            cost_result = cost_agent.estimate_with_ml(mission_req.__dict__)
        else:
            # Fallback to simple estimation
            cost_result = {
                'total_estimated_cost': 500_000_000,
                'confidence_interval_90': (425_000_000, 575_000_000),
                'wbs_breakdown': {
                    'spacecraft_development': 175_000_000,
                    'launch_services': 125_000_000,
                    'operations': 75_000_000,
                    'ground_segment': 50_000_000,
                    'project_management': 40_000_000,
                    'contingency': 35_000_000
                }
            }
        
        # Risk assessment
        risk_result = {
            'identified_risks': [
                {
                    'description': 'Technology development delays',
                    'category': 'Technical',
                    'probability': 0.3,
                    'impact': 50_000_000
                },
                {
                    'description': 'Launch vehicle availability',
                    'category': 'Schedule',
                    'probability': 0.2,
                    'impact': 30_000_000
                },
                {
                    'description': 'Cost overruns in development',
                    'category': 'Cost',
                    'probability': 0.4,
                    'impact': 60_000_000
                }
            ],
            'risk_score': 35.5
        }
        
        # Optimization
        optimization_result = {
            'optimized_allocation': {
                'spacecraft_development': 165_000_000,
                'launch_services': 120_000_000,
                'operations': 70_000_000,
                'ground_segment': 47_000_000,
                'project_management': 38_000_000,
                'contingency': 40_000_000  # Increased due to risks
            },
            'expected_savings': 20_000_000,
            'savings_percentage': 4.0
        }
        
        # Compile results
        results = {
            'mission_id': mission_req.mission_id,
            'cost_analysis': cost_result,
            'risk_analysis': risk_result,
            'optimization': optimization_result,
            'confidence_score': 0.82
        }
        
        # Generate visualizations
        print("\n📊 Generating visualization dashboard...")
        dashboard = self.viz_engine.create_dashboard(results)
        
        # Add training data visualization if available
        if self.training_data is not None and self.trained_models.get('distributions'):
            dashboard['distributions'] = self.viz_engine.create_distribution_plots(
                self.training_data, 
                self.trained_models['distributions']
            )
        
        print("✓ Analysis complete with visualizations")
        
        return results, dashboard

# ============================================================================
# MAIN EXECUTION WITH VISUALIZATION
# ============================================================================

def run_enhanced_analysis(csv_path: str = None, mission: MissionRequirements = None):
    """
    Run enhanced analysis with data ingestion and visualization
    
    Args:
        csv_path: Path to CSV file with training data (optional)
        mission: Mission requirements (optional)
    
    Returns:
        Tuple of (results, visualizations)
    """
    
    # Initialize orchestrator
    orchestrator = EnhancedAgentOrchestrator()
    
    # Load and train from CSV if provided
    if csv_path:
        dist_fig = orchestrator.load_and_train(csv_path)
        if dist_fig:
            print("Distributions learned from data")
    else:
        # Use synthetic data
        print("No CSV provided, generating synthetic training data...")
        orchestrator.load_and_train(None)
    
    # Create sample mission if not provided
    if mission is None:
        mission = MissionRequirements(
            mission_id="DEMO-2025-001",
            mission_type="Mars",
            duration_months=36,
            payload_mass_kg=7500,
            launch_date=datetime(2028, 7, 15),
            technology_readiness_level=6,
            crew_size=0,
            science_objectives=["Surface analysis", "Atmospheric sampling"],
            constraints={"max_launch_mass": 10000}
        )
    
    # Process mission with visualization
    results, dashboard = orchestrator.process_mission_with_visualization(mission)
    
    # Save visualizations as high-resolution PNGs
    orchestrator.viz_engine.save_all_figures(dashboard)
    
    return results, dashboard

# ============================================================================
# JUPYTER HELPER FUNCTIONS
# ============================================================================

def load_mission_from_csv_row(df: pd.DataFrame, row_index: int = 0) -> MissionRequirements:
    """Helper function to create mission from CSV row"""
    if row_index >= len(df):
        print(f"Warning: Row {row_index} not found, using first row")
        row_index = 0
    
    row = df.iloc[row_index]
    return MissionRequirements.from_dataframe_row(row)

def batch_process_missions(csv_path: str, output_dir: str = "./batch_results"):
    """Process multiple missions from CSV and save results"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    pipeline = DataIngestionPipeline()
    df = pipeline.load_csv_data(csv_path) if csv_path else pipeline.generate_synthetic_data()
    
    # Initialize orchestrator
    orchestrator = EnhancedAgentOrchestrator()
    orchestrator.load_and_train(csv_path)
    
    results_list = []
    
    # Process each mission
    for idx in range(min(10, len(df))):  # Process up to 10 missions
        mission = load_mission_from_csv_row(df, idx)
        results, dashboard = orchestrator.process_mission_with_visualization(mission)
        
        # Save individual results
        with open(f"{output_dir}/mission_{mission.mission_id}_results.json", 'w') as f:
            json.dump(results, f, default=str, indent=2)
        
        results_list.append(results)
        
        print(f"✓ Processed mission {idx+1}/{min(10, len(df))}: {mission.mission_id}")
    
    print(f"\n✓ Batch processing complete. Results saved to {output_dir}/")
    return results_list

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("""
    ========================================
    ENHANCED SPACE MISSION FINANCIAL PLANNING
    With CSV Data Ingestion & Visualization
    ========================================
    
    Usage Examples:
    
    1. Basic analysis with synthetic data:
       results, charts = run_enhanced_analysis()
    
    2. Load and train from CSV:
       results, charts = run_enhanced_analysis('mission_data.csv')
    
    3. Batch process multiple missions:
       results = batch_process_missions('missions.csv')
    
    4. Custom mission from CSV row:
       df = pd.read_csv('missions.csv')
       mission = load_mission_from_csv_row(df, row_index=5)
       results, charts = run_enhanced_analysis(mission=mission)
    """)
    
    # Run demo with synthetic data
    results, dashboard = run_enhanced_analysis()
    
    print("\n✓ Demo complete! Check the generated visualizations.")
