from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import openai
import os
from tableau_api_lib import TableauServerConnection
from tableau_api_lib.utils.querying import get_views_dataframe, get_view_data_dataframe
import json
from datetime import datetime
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import base64
import io
import numpy as np
from collections import Counter
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class TableauChatBot:
    def __init__(self):
        self.tableau_config = {
            'my_env': {
                'server': 'https://us-west-2b.online.tableau.com',
                'site_name': 'Site Name Comes here',
                'site_url': 'Site URL Name Comes here',
                'api_version': '3.26',
                'personal_access_token_name': 'APIToken',
                'personal_access_token_secret': 'Tableau API Token comes here'
            }
        }
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key='Open AI Token Comes here'
        )
        
        # CONFIGURE ALLOWED REPORTS HERE
        # Option 1: Filter by report names (exact match)
        self.allowed_report_names = [
            'Claims Report',
            'Report for Demo',
            # Add your specific report names here
        ]
        
        # Option 2: Filter by keywords in report names
        self.allowed_keywords = [            
            'Demo',
            # Add keywords that should be included
        ]
        
        # Option 3: Filter by specific view IDs
        self.allowed_view_ids = [
            # 'view-id-1',
            # 'view-id-2',
            # Add specific view IDs if you know them
        ]
        
        # Option 4: Exclude certain reports
        self.excluded_keywords = [            
            'draft',
            'temp',
            'backup',
            'dashboard',
            'Alaska AUTO summary all directions',
            'sales',
            'Print Page',
            'Latest',
            # Add keywords to exclude
        ]
        
        # Choose filtering method: 'names', 'keywords', 'view_ids', 'exclude', 'combined', or 'all'
        self.filter_method = 'keywords'  # Change this to your preferred method
        
        # Duplicate handling options
        self.duplicate_preference = 'first'  # Options: 'first', 'last', 'most_recent', 'alphabetical'
        
        self.tableau_conn = None
        self.available_views = None
        self.cached_data = {}
        
        # Set up plotting style
        # plt.style.use('seaborn-v0_8')  # Comment out if causing issues
        sns.set_palette("husl")
        
    def connect_to_tableau(self):
        """Establish connection to Tableau Server"""
        try:
            self.tableau_conn = TableauServerConnection(
                config_json=self.tableau_config, 
                env='my_env'
            )
            self.tableau_conn.sign_in()
            logger.info("Successfully connected to Tableau Server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Tableau: {str(e)}")
            return False
    
    def filter_views(self, views_df):
        """Filter views based on configuration and remove duplicates by name"""
        if views_df is None or views_df.empty:
            return []
            
        # Convert to list of dictionaries for easier processing
        all_views = views_df[['id', 'name', 'contentUrl', 'createdAt', 'updatedAt']].to_dict('records')
        filtered_views = []
        
        # First pass: Apply content filtering
        content_filtered = []
        for view in all_views:
            view_name = view['name'].lower()
            view_id = view['id']
            original_name = view['name']
            
            # Apply filtering based on selected method
            should_include = False
            
            if self.filter_method == 'names':
                if view['name'] in self.allowed_report_names:
                    should_include = True
            elif self.filter_method == 'keywords':
                if any(keyword.lower() in view_name for keyword in self.allowed_keywords):
                    should_include = True
            elif self.filter_method == 'view_ids':
                if view_id in self.allowed_view_ids:
                    should_include = True
            elif self.filter_method == 'exclude':
                if not any(keyword.lower() in view_name for keyword in self.excluded_keywords):
                    should_include = True
            elif self.filter_method == 'all':
                should_include = True
            elif self.filter_method == 'combined':
                include_by_keyword = any(keyword.lower() in view_name for keyword in self.allowed_keywords)
                exclude_by_keyword = any(keyword.lower() in view_name for keyword in self.excluded_keywords)
                if include_by_keyword and not exclude_by_keyword:
                    should_include = True
            
            if should_include:
                content_filtered.append(view)
        
        # Second pass: Remove duplicates by name
        name_groups = {}
        for view in content_filtered:
            name = view['name']
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(view)
        
        # Select one view per unique name based on preference
        for name, views_with_same_name in name_groups.items():
            if len(views_with_same_name) == 1:
                filtered_views.append(views_with_same_name[0])
            else:
                # Handle duplicates based on preference
                selected_view = self._select_preferred_duplicate(views_with_same_name)
                filtered_views.append(selected_view)
                logger.info(f"Found {len(views_with_same_name)} duplicates for '{name}', selected: {selected_view['id']}")
        
        logger.info(f"Filtered {len(all_views)} views down to {len(filtered_views)} unique views using method: {self.filter_method}")
        logger.info(f"Removed {len(content_filtered) - len(filtered_views)} duplicate reports")
        
        return filtered_views
    
    def _select_preferred_duplicate(self, duplicate_views):
        """Select preferred view when multiple views have the same name"""
        try:
            if self.duplicate_preference == 'first':
                return duplicate_views[0]
            elif self.duplicate_preference == 'last':
                return duplicate_views[-1]
            elif self.duplicate_preference == 'most_recent':
                # Sort by updatedAt if available, otherwise createdAt
                sorted_views = sorted(duplicate_views, 
                    key=lambda x: x.get('updatedAt', x.get('createdAt', '')), 
                    reverse=True)
                return sorted_views[0]
            elif self.duplicate_preference == 'alphabetical':
                # Sort by contentUrl or id for consistent ordering
                sorted_views = sorted(duplicate_views, 
                    key=lambda x: x.get('contentUrl', x.get('id', '')))
                return sorted_views[0]
            else:
                return duplicate_views[0]  # Default to first
        except Exception as e:
            logger.warning(f"Error selecting preferred duplicate: {str(e)}, using first")
            return duplicate_views[0]
            
    def get_available_views(self):
        """Fetch all available Tableau views and apply filtering"""
        try:
            if not self.tableau_conn:
                if not self.connect_to_tableau():
                    return None
                    
            self.available_views = get_views_dataframe(self.tableau_conn)
            logger.info(f"Retrieved {len(self.available_views)} total views from Tableau")
            
            # Apply filtering
            filtered_views = self.filter_views(self.available_views)
            
            # Log filtered results
            if filtered_views:
                logger.info("Filtered views:")
                for view in filtered_views:
                    logger.info(f"  - {view['name']} (ID: {view['id']})")
            else:
                logger.warning("No views match the current filter criteria")
            
            return filtered_views
            
        except Exception as e:
            logger.error(f"Error fetching views: {str(e)}")
            return None
    
    def get_view_data(self, view_id, force_refresh=False):
        """Fetch data for a specific view"""
        try:
            logger.info(f"Getting view data for view_id: {view_id} | force_refresh={force_refresh}")

            # Check cache first
            if view_id in self.cached_data and not force_refresh:
                logger.info(f"Using cached data for view {view_id}")
                return self.cached_data[view_id]
            
            if not self.tableau_conn:
                if not self.connect_to_tableau():
                    logger.error("Cannot connect to Tableau Server")
                    return None
            
            # Fetch fresh data with error handling
            try:
                df = get_view_data_dataframe(self.tableau_conn, view_id=view_id)
                logger.info(f"DataFrame shape: {df.shape}")
                logger.debug(f"Sample data: {df.head().to_dict()}")
                
                # Handle empty dataframes
                if df.empty:
                    logger.warning(f"Empty dataframe returned for view {view_id}")
                    return {
                        'data': df,
                        'timestamp': datetime.now(),
                        'record_count': 0,
                        'columns': list(df.columns) if not df.empty else []
                    }
                
            except Exception as data_error:
                logger.error(f"Error fetching data from Tableau API: {str(data_error)}")
                return None
            
            # Cache the data
            self.cached_data[view_id] = {
                'data': df,
                'timestamp': datetime.now(),
                'record_count': len(df),
                'columns': list(df.columns)
            }
            
            logger.info(f"Retrieved {len(df)} records for view {view_id}")
            return self.cached_data[view_id]
            
        except Exception as e:
            logger.error(f"Error fetching view data: {str(e)}")
            return None

    def detect_chart_intent(self, user_question, df):
        """Simplified chart detection - only bar, line, and pie"""
        try:
            question_lower = user_question.lower()
            logger.info(f"CHART_DEBUG: Analyzing question: '{question_lower}'")
            
            # Check if user wants any chart
            chart_triggers = ['chart', 'graph', 'plot', 'visualize', 'show', 'display']
            wants_chart = any(trigger in question_lower for trigger in chart_triggers)
            
            if not wants_chart:
                return False, None
            
            # Determine chart type based on keywords
            if any(word in question_lower for word in ['trend', 'over time', 'timeline', 'monthly', 'yearly']):
                return True, 'line'
            elif any(word in question_lower for word in ['proportion', 'percentage', 'distribution', 'share']):
                return True, 'pie'
            else:
                return True, 'bar'  # Default to bar chart
                
        except Exception as e:
            logger.error(f"Error in chart detection: {str(e)}")
            return False, None

    def _convert_to_numeric(self, series):
        """Enhanced numeric conversion with detailed logging"""
        try:
            logger.info(f"CHART_DEBUG: Converting series to numeric - original type: {series.dtype}")
            logger.info(f"CHART_DEBUG: Sample values before conversion: {series.head().tolist()}")
            
            # Try direct conversion first
            numeric_result = pd.to_numeric(series, errors='coerce')
            valid_count = numeric_result.notna().sum()
            
            logger.info(f"CHART_DEBUG: Direct conversion - {valid_count} valid values")
            
            if valid_count > 0:
                logger.info(f"CHART_DEBUG: Direct conversion successful")
                return numeric_result
            
            # Try cleaning and converting
            logger.info(f"CHART_DEBUG: Trying cleaned conversion...")
            cleaned = series.astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '').str.strip()
            cleaned_result = pd.to_numeric(cleaned, errors='coerce')
            cleaned_valid = cleaned_result.notna().sum()
            
            logger.info(f"CHART_DEBUG: Cleaned conversion - {cleaned_valid} valid values")
            
            if cleaned_valid > 0:
                return cleaned_result
            
            logger.info(f"CHART_DEBUG: No numeric conversion possible")
            return series
            
        except Exception as e:
            logger.error(f"CHART_DEBUG: Error in numeric conversion: {str(e)}")
            return series

    def _get_chart_columns(self, df, user_question):
        """Smart column detection with fuzzy matching"""
        try:
            question_lower = user_question.lower()
            logger.info(f"CHART_DEBUG: Question analysis: '{question_lower}'")
            logger.info(f"CHART_DEBUG: Available columns: {list(df.columns)}")
            
            potential_x_col = None
            potential_y_col = None
            
            # Create a mapping of keywords to actual column names
            column_matches = {}
            for col in df.columns:
                col_lower = col.lower()
                # Split column names by common separators
                col_words = col_lower.replace('_', ' ').replace('-', ' ').split()
                column_matches[col] = col_words + [col_lower]
            
            logger.info(f"CHART_DEBUG: Column word mapping: {column_matches}")
            
            # Look for column references in the question
            question_words = question_lower.replace('_', ' ').replace('-', ' ').split()
            
            for col, search_terms in column_matches.items():
                for term in search_terms:
                    if term in question_lower:
                        logger.info(f"CHART_DEBUG: Found match '{term}' for column '{col}'")
                        if potential_x_col is None:
                            potential_x_col = col
                        elif potential_y_col is None and col != potential_x_col:
                            potential_y_col = col
            
            # Try to detect based on common patterns
            if not potential_x_col or not potential_y_col:
                # Look for company/customer/name columns (typically X-axis)
                for col in df.columns:
                    col_lower = col.lower()
                    if any(word in col_lower for word in ['company', 'customer', 'name', 'category', 'type']):
                        if not potential_x_col:
                            potential_x_col = col
                            logger.info(f"CHART_DEBUG: Auto-detected X column: {col}")
                            break
                
                # Look for numeric-like columns (typically Y-axis)
                for col in df.columns:
                    col_lower = col.lower()
                    if any(word in col_lower for word in ['count', 'total', 'sum', 'amount', 'value', 'price', 'revenue', 'quantity']):
                        if col != potential_x_col and not potential_y_col:
                            potential_y_col = col
                            logger.info(f"CHART_DEBUG: Auto-detected Y column: {col}")
                            break
            
            logger.info(f"CHART_DEBUG: Final detected columns - X: {potential_x_col}, Y: {potential_y_col}")
            return potential_x_col, potential_y_col
            
        except Exception as e:
            logger.error(f"Error detecting columns: {str(e)}")
            return None, None

    def make_json_serializable(self, obj):
        """Simple JSON serialization"""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, dict):
            return {key: self.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def create_visualization(self, df, chart_type, user_question):
        """Enhanced chart creation with better debugging"""
        try:
            logger.info(f"CHART_DEBUG: === Starting chart creation ===")
            logger.info(f"CHART_DEBUG: Chart type: {chart_type}")
            logger.info(f"CHART_DEBUG: DataFrame shape: {df.shape}")
            logger.info(f"CHART_DEBUG: DataFrame columns: {list(df.columns)}")
            logger.info(f"CHART_DEBUG: User question: {user_question}")
            
            if df.empty:
                logger.error("CHART_DEBUG: DataFrame is empty!")
                return None, "No data available"
            
            # Get column suggestions
            suggested_x, suggested_y = self._get_chart_columns(df, user_question)
            logger.info(f"CHART_DEBUG: Suggested columns - X: {suggested_x}, Y: {suggested_y}")
            
            # Create chart based on type
            if chart_type == 'bar':
                result = self._create_smart_bar_chart(df, suggested_x, suggested_y, user_question)
            elif chart_type == 'line':
                result = self._create_smart_line_chart(df, suggested_x, suggested_y, user_question)
            elif chart_type == 'pie':
                result = self._create_smart_pie_chart(df, suggested_x or suggested_y, user_question)
            else:
                result = self._create_smart_bar_chart(df, suggested_x, suggested_y, user_question)
            
            logger.info(f"CHART_DEBUG: Chart creation result: {result[0] is not None}")
            if result[0] is None:
                logger.error(f"CHART_DEBUG: Chart creation failed: {result[1]}")
            else:
                logger.info(f"CHART_DEBUG: Chart created successfully: {result[1]}")
            
            return result
            
        except Exception as e:
            logger.error(f"CHART_DEBUG: Exception in create_visualization: {str(e)}")
            import traceback
            logger.error(f"CHART_DEBUG: Traceback: {traceback.format_exc()}")
            return None, f"Chart creation failed: {str(e)}"

    def _create_smart_bar_chart(self, df, suggested_x, suggested_y, user_question):
        """Enhanced bar chart creation with extensive debugging"""
        try:
            logger.info(f"CHART_DEBUG: === Bar Chart Creation ===")
            logger.info(f"CHART_DEBUG: Suggested X: {suggested_x}, Y: {suggested_y}")
            
            # Get column types
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            all_cols = df.columns.tolist()
            
            logger.info(f"CHART_DEBUG: Text columns: {text_cols}")
            logger.info(f"CHART_DEBUG: Numeric columns: {numeric_cols}")
            
            # Determine final columns
            x_col = suggested_x
            y_col = suggested_y
            
            # If no suggestions, use smart defaults
            if not x_col:
                # Prefer text columns for X-axis
                x_col = text_cols[0] if text_cols else all_cols[0]
                logger.info(f"CHART_DEBUG: No X suggestion, using: {x_col}")
            
            if not y_col:
                # Try to find a good Y column
                remaining_cols = [col for col in all_cols if col != x_col]
                
                # First try numeric columns
                numeric_options = [col for col in remaining_cols if col in numeric_cols]
                if numeric_options:
                    y_col = numeric_options[0]
                    logger.info(f"CHART_DEBUG: Found numeric Y column: {y_col}")
                else:
                    # Try to convert text columns to numeric
                    for col in remaining_cols:
                        if col in text_cols:
                            logger.info(f"CHART_DEBUG: Testing if {col} can be converted to numeric...")
                            test_conversion = self._convert_to_numeric(df[col])
                            valid_count = test_conversion.notna().sum()
                            logger.info(f"CHART_DEBUG: Column {col} - {valid_count} valid numeric values out of {len(df)}")
                            if valid_count > 0:
                                y_col = col
                                logger.info(f"CHART_DEBUG: Selected convertible Y column: {y_col}")
                                break
                    
                    # Last resort
                    if not y_col:
                        y_col = remaining_cols[0] if remaining_cols else x_col
                        logger.info(f"CHART_DEBUG: Last resort Y column: {y_col}")
            
            logger.info(f"CHART_DEBUG: Final columns - X: {x_col}, Y: {y_col}")
            
            # Validate columns exist
            if x_col not in df.columns:
                logger.error(f"CHART_DEBUG: X column '{x_col}' not found in dataframe!")
                return None, f"Column '{x_col}' not found"
            
            if y_col not in df.columns:
                logger.error(f"CHART_DEBUG: Y column '{y_col}' not found in dataframe!")
                return None, f"Column '{y_col}' not found"
            
            # Prepare data
            logger.info(f"CHART_DEBUG: Preparing chart data...")
            chart_data = df[[x_col, y_col]].copy()
            logger.info(f"CHART_DEBUG: Initial data shape: {chart_data.shape}")
            
            # Show sample of original data
            logger.info(f"CHART_DEBUG: Original Y column sample: {chart_data[y_col].head().tolist()}")
            logger.info(f"CHART_DEBUG: Original Y column type: {chart_data[y_col].dtype}")
            
            # Convert Y column to numeric
            original_y_type = chart_data[y_col].dtype
            chart_data[y_col + '_numeric'] = self._convert_to_numeric(chart_data[y_col])
            
            # Check conversion results
            valid_numeric = chart_data[y_col + '_numeric'].notna().sum()
            logger.info(f"CHART_DEBUG: Numeric conversion - {valid_numeric} valid values out of {len(chart_data)}")
            logger.info(f"CHART_DEBUG: Converted Y sample: {chart_data[y_col + '_numeric'].dropna().head().tolist()}")
            
            if valid_numeric == 0:
                logger.error(f"CHART_DEBUG: No valid numeric data in column {y_col}")
                return None, f"Column '{y_col}' contains no convertible numeric data"
            
            # Use converted column and remove invalid rows
            chart_data = chart_data.dropna(subset=[y_col + '_numeric'])
            chart_data[y_col] = chart_data[y_col + '_numeric']
            chart_data = chart_data.drop(columns=[y_col + '_numeric'])
            
            logger.info(f"CHART_DEBUG: Data after numeric conversion: {chart_data.shape}")
            
            # Group and aggregate if X is categorical
            if chart_data[x_col].dtype == 'object' or len(chart_data[x_col].unique()) < len(chart_data) * 0.5:
                logger.info(f"CHART_DEBUG: Grouping by {x_col} and summing {y_col}")
                grouped = chart_data.groupby(x_col)[y_col].sum().reset_index()
                logger.info(f"CHART_DEBUG: Grouped data shape: {grouped.shape}")
            else:
                logger.info(f"CHART_DEBUG: Using data as-is (no grouping)")
                grouped = chart_data.head(20)  # Limit for performance
            
            # Sort and limit
            grouped = grouped.sort_values(y_col, ascending=False).head(15)
            logger.info(f"CHART_DEBUG: Final chart data shape: {grouped.shape}")
            logger.info(f"CHART_DEBUG: Final X values: {grouped[x_col].tolist()[:5]}...")
            logger.info(f"CHART_DEBUG: Final Y values: {grouped[y_col].tolist()[:5]}...")
            
            # Create Plotly chart
            logger.info(f"CHART_DEBUG: Creating Plotly chart...")
            fig = {
                'data': [{
                    'type': 'bar',
                    'x': grouped[x_col].astype(str).tolist(),
                    'y': grouped[y_col].tolist(),
                    'name': y_col,
                    'marker': {'color': 'rgba(55, 128, 191, 0.7)'}
                }],
                'layout': {
                    'title': f'{y_col} by {x_col}',
                    'xaxis': {'title': x_col, 'tickangle': -45},
                    'yaxis': {'title': y_col},
                    'height': 400,
                    'margin': {'l': 50, 'r': 50, 't': 50, 'b': 100}
                }
            }
            
            logger.info(f"CHART_DEBUG: Chart created successfully!")
            return fig, f"Bar chart showing {y_col} by {x_col} (converted from {original_y_type} to numeric)"
            
        except Exception as e:
            logger.error(f"CHART_DEBUG: Exception in bar chart creation: {str(e)}")
            import traceback
            logger.error(f"CHART_DEBUG: Full traceback: {traceback.format_exc()}")
            return None, f"Bar chart error: {str(e)}"

    def _create_smart_pie_chart(self, df, suggested_col, user_question):
        """Create pie chart with smart column selection"""
        try:
            # Use suggested column or first text column
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            col = suggested_col if suggested_col in df.columns else (text_cols[0] if text_cols else df.columns[0])
            
            # Count values
            value_counts = df[col].value_counts().head(8)
            
            if value_counts.empty:
                return None, "No data for pie chart"
            
            fig = {
                'data': [{
                    'type': 'pie',
                    'labels': value_counts.index.astype(str).tolist(),
                    'values': value_counts.values.tolist(),
                    'name': col
                }],
                'layout': {
                    'title': f'Distribution of {col}',
                    'height': 400
                }
            }
            
            return fig, f"Pie chart: Distribution of {col}"
            
        except Exception as e:
            logger.error(f"Error in pie chart: {str(e)}")
            return None, f"Pie chart error: {str(e)}"

    def _create_smart_line_chart(self, df, suggested_x, suggested_y, user_question):
        """Create line chart with smart column selection"""
        try:
            x_col = suggested_x or df.columns[0]
            y_col = suggested_y or df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            # Ensure y_col is numeric
            chart_data = df[[x_col, y_col]].copy()
            chart_data[y_col] = self._convert_to_numeric(chart_data[y_col])
            chart_data = chart_data.dropna()
            
            if chart_data.empty:
                return None, "No valid numeric data for line chart"
            
            # Sort by x column
            if chart_data[x_col].dtype in ['datetime64[ns]', 'object']:
                chart_data = chart_data.sort_values(x_col)
            
            # Limit data points
            if len(chart_data) > 50:
                chart_data = chart_data.head(50)
            
            fig = {
                'data': [{
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'x': chart_data[x_col].astype(str).tolist(),
                    'y': chart_data[y_col].tolist(),
                    'name': y_col,
                    'line': {'color': 'rgba(55, 128, 191, 1)'}
                }],
                'layout': {
                    'title': f'{y_col} over {x_col}',
                    'xaxis': {'title': x_col},
                    'yaxis': {'title': y_col},
                    'height': 400
                }
            }
            
            return fig, f"Line chart: {y_col} over {x_col}"
            
        except Exception as e:
            logger.error(f"Error in line chart: {str(e)}")
            return None, f"Line chart error: {str(e)}"

     
    def analyze_data_with_ai(self, view_id, user_question, context_data=None):
        """Use OpenAI to analyze the data and answer user questions"""
        try:
            # Get the data
            view_data = self.get_view_data(view_id)
            if not view_data:
                logger.warning(f"No data returned for view_id: {view_id}")
                return {"response": "Sorry, I couldn't retrieve the data for analysis. Please check if the view exists and you have proper permissions."}
            
            df = view_data['data']
            
            # Handle empty dataframes
            if df.empty:
                return {"response": f"The selected view appears to be empty (0 records). Please check if the view contains data or select a different report."}
            
            # Check if user wants a visualization
            wants_chart, chart_type = self.detect_chart_intent(user_question, df)
            
            chart_data = None
            chart_description = None
            
            if wants_chart:
                logger.info(f"CHART_DEBUG: Chart requested - type: {chart_type}")
                chart_data, chart_description = self.create_visualization(df, chart_type, user_question)
                if chart_data:
                    logger.info(f"CHART_DEBUG: Chart successfully created: {chart_description}")
                else:
                    logger.warning(f"CHART_DEBUG: Chart creation failed: {chart_description}")
            
            # Prepare highlighted column headers info
            column_headers_info = self._format_column_headers(df)
            
            # Prepare ACTUAL data summary and analysis
            try:
                data_insights = self._prepare_detailed_analysis(df, user_question)
                data_summary = self._prepare_data_summary(df)
            except Exception as summary_error:
                logger.error(f"Error preparing data analysis: {str(summary_error)}")
                data_insights = f"Data contains {len(df)} records with {len(df.columns)} columns: {', '.join(df.columns)}"
                data_summary = "Basic data summary available"
            
            # Create context-aware prompt with EXPLICIT instructions
            system_prompt = self._create_detailed_system_prompt(view_data, data_summary, wants_chart)
            
            # Prepare user prompt with ACTUAL data values (not placeholders)
            sample_data_text = self._format_sample_data(df, limit=10)
            
            chart_context = ""
            if wants_chart and chart_data:
                chart_context = f"\nCHART CREATED: {chart_description}\nThe user will see this visualization along with your response."
            elif wants_chart and not chart_data:
                chart_context = f"\nCHART REQUEST: User requested a chart but it couldn't be created due to data limitations: {chart_description}"
            
            user_prompt = f"""IMPORTANT: Use ONLY the actual data values provided below. Do NOT use placeholders like [Customer Name] or [Revenue Value].

            User Question: {user_question}
            
            {column_headers_info}
            
            ACTUAL DATA INSIGHTS:
            {data_insights}
            
            DATA SUMMARY:
            {data_summary}
            
            ACTUAL SAMPLE DATA:
            {sample_data_text}
            
            {chart_context}
            
            CRITICAL FORMATTING INSTRUCTIONS - FOLLOW EXACTLY:
            1. Answer using ONLY the real data values shown above
            2. Include specific numbers, names, and values from the actual data
            3. Do NOT use any placeholders or bracketed values like [Value] or [Name]
            4. If you cannot find specific data to answer the question, say so explicitly
            5. Provide concrete examples using the real data values
            6. When referencing data fields, use the highlighted column headers shown above
            7. If a chart was created, acknowledge it and explain what insights it reveals
            
            MANDATORY RESPONSE FORMAT - USE THIS EXACT STRUCTURE:
            
            **[Section Title]:**
            
            1. First numbered item
            2. Second numbered item  
            3. Third numbered item
            
            **[Next Section Title]:**
            
            - First bullet point
            - Second bullet point
            - Third bullet point
            
            EXAMPLE OF EXPECTED OUTPUT FORMAT:
            
            **Sales Person List:**
            
            **Destination Sales Persons:**
            1. Chris Olsen
            2. Jim Trench
            3. Sheila Hubbard
            4. Chris Brown
            5. Max Folk
            
            **Origin Sales Persons:**
            1. Chris Olsen  
            2. Jim Trench
            3. Sheila Hubbard
            4. Chris Brown
            5. Max Folk
            
            **Key Insights:**
            
            - There are 5 unique sales persons in each category
            - Sales persons handle both destination and origin roles
            - The dataset shows consistent coverage across territories
            
            **Recommendations:**
            
            1. Analyze individual performance metrics
            2. Provide targeted training programs
            3. Monitor sales effectiveness regularly
            
            NOW ANSWER THE USER'S QUESTION USING THIS EXACT FORMAT WITH PROPER LINE BREAKS AND STRUCTURE."""
            
            # Make API call to OpenAI with error handling
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1500,  # Increased for more detailed responses
                    temperature=0.3   # Lower temperature for more factual responses
                )
                
                ai_response = response.choices[0].message.content

                # FORMAT THE RESPONSE PROPERLY
                # formatted_response = self.format_ai_response(ai_response)

                # FORCE CLEAN THE RESPONSE
                cleaned_response = self.clean_ai_response(ai_response)

                logger.info(f"RESPONSE_DEBUG: Original AI response length: {len(ai_response)}")
                logger.info(f"RESPONSE_DEBUG: Cleaned response length: {len(cleaned_response)}")
                logger.info(f"RESPONSE_DEBUG: First 300 chars of cleaned: {cleaned_response[:300]}")
                
                # Return response with optional chart data
                result = {"response": cleaned_response}
                if chart_data:
                    result["chart"] = chart_data
                    result["chart_description"] = chart_description
                    logger.info(f"CHART_DEBUG: Returning response with chart data")
                
                return result
                
            except openai.APIError as api_error:
                logger.error(f"OpenAI API error: {str(api_error)}")
                return {"response": f"Sorry, I encountered an issue with the AI service. Please try again later. Error: {str(api_error)}"}
            except Exception as openai_error:
                logger.error(f"OpenAI processing error: {str(openai_error)}")
                return {"response": f"Sorry, I encountered an error while processing your question with AI. Please try rephrasing your question."}
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return {"response": f"Sorry, I encountered an error while analyzing the data: {str(e)}"}

    def clean_ai_response(self, response_text):
        """Force clean formatting on AI response regardless of how it was generated"""
        try:
            logger.info(f"CLEAN_DEBUG: Original response: {response_text[:200]}...")
            
            # Start fresh
            cleaned = response_text
            
            # Step 1: Fix section headers - add line breaks around **Text:**
            cleaned = re.sub(r'(\*\*[^*]+:\*\*)', r'\n\n\1\n', cleaned)
            
            # Step 2: Fix numbered lists - ensure each number starts on new line
            cleaned = re.sub(r'(\d+\.\s+)', r'\n\1', cleaned)
            
            # Step 3: Fix bullet points - convert "- " to proper bullets on new lines
            cleaned = re.sub(r'\s+-\s+', '\n- ', cleaned)
            
            # Step 4: Clean up multiple consecutive line breaks
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
            
            # Step 5: Remove leading/trailing whitespace
            cleaned = cleaned.strip()
            
            # Step 6: Ensure there's space after colons in headers
            cleaned = re.sub(r'(\*\*[^*]+):\*\*', r'\1:**', cleaned)
            
            logger.info(f"CLEAN_DEBUG: Cleaned response: {cleaned[:200]}...")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning response: {str(e)}")
            return response_text
        
    def _prepare_data_summary(self, df):
        """Create a comprehensive data summary for AI analysis"""
        try:
            # Handle empty dataframes
            if df.empty:
                return "Dataset is empty (0 records)"
            
            summary = f"""
            Dataset Overview:
            - Total Records: {len(df):,}
            - Total Columns: {len(df.columns)}
            - Column Names: {', '.join(df.columns)}
            """
            
            # Add data types safely
            try:
                summary += f"\nData Types:\n{df.dtypes.to_string()}"
            except Exception:
                summary += f"\nData Types: Unable to determine"
            
            # Add numerical summary safely
            try:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    summary += f"\nNumerical Columns Summary:\n{df.describe().to_string()}"
                else:
                    summary += f"\nNumerical Columns Summary: No numerical columns found"
            except Exception:
                summary += f"\nNumerical Columns Summary: Unable to compute"
            
            # Add missing values info safely
            try:
                summary += f"\nMissing Values:\n{df.isnull().sum().to_string()}"
            except Exception:
                summary += f"\nMissing Values: Unable to compute"
            
            # Add sample values for each column
            summary += f"\nSample Values per Column:"
            for col in df.columns:
                try:
                    unique_vals = df[col].dropna().unique()[:3]  # Reduced to 3 values
                    summary += f"\n{col}: {', '.join(map(str, unique_vals))}"
                    if len(df[col].unique()) > 3:
                        summary += f" (and {len(df[col].unique()) - 3} more...)"
                except Exception:
                    summary += f"\n{col}: Unable to get sample values"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating data summary: {str(e)}")
            return f"Data summary: {len(df)} records with {len(df.columns)} columns"
    
    def _format_column_headers(self, df, view_info=None):
        """Format only Text/Category Fields and Dataset Summary as footer with report link"""
        try:
            if df.empty or len(df.columns) == 0:
                return ""
            
            # Get text/categorical columns
            text_cols = []
            for col in df.columns:
                dtype = str(df[col].dtype).lower()
                if 'object' in dtype or 'string' in dtype:
                    text_cols.append(col)
            
            # Format the footer information
            footer_info = []
            
            # Add report link if available
            if view_info and 'name' in view_info and 'id' in view_info:
                # Construct Tableau Online URL
                server_url = self.tableau_config['my_env']['server']
                site_name = self.tableau_config['my_env']['site_name']
                view_id = view_info['id']
                
                # Tableau Online view URL format
                report_url = f"{server_url}/#/site/{site_name}/views/{view_id}"
                
                footer_info.append(f"🔗 **Have full Tableau Access? View the original Tableau report here :** [View in Tableau]({report_url})")
            
            # Add text/categorical fields as comma-separated list
            if text_cols:
                text_fields_list = []
                for col in text_cols:
                    try:
                        unique_count = df[col].nunique()
                        text_fields_list.append(f"**{col}**")
                    except:
                        text_fields_list.append(f"**{col}**")
                
                footer_info.append(f"📝 **Report Fields:** {', '.join(text_fields_list)}")
            
            # Join with line breaks
            return "\n".join(footer_info)
            
        except Exception as e:
            logger.error(f"Error formatting column headers: {str(e)}")
            return f"📈 **Dataset Summary:** {len(df)} records, {len(df.columns)} fields"

    def _prepare_detailed_analysis(self, df, user_question):
        """Prepare detailed analysis of the actual data"""
        try:
            analysis = []
            
            # Basic data info
            analysis.append(f"Dataset contains {len(df):,} records with {len(df.columns)} columns")
            
            # Column information with actual sample values
            analysis.append("\nCOLUMN DETAILS WITH ACTUAL VALUES:")
            for col in df.columns:
                try:
                    # Get actual non-null values
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # For numeric columns
                        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                            analysis.append(f"- {col}: Numeric values from {non_null_values.min()} to {non_null_values.max()}")
                            analysis.append(f"  Sample values: {list(non_null_values.head(3))}")
                        else:
                            # For text/categorical columns
                            unique_vals = non_null_values.unique()[:5]
                            analysis.append(f"- {col}: Text/Category with {len(df[col].unique())} unique values")
                            analysis.append(f"  Actual values: {list(unique_vals)}")
                    else:
                        analysis.append(f"- {col}: All values are null/empty")
                except Exception:
                    analysis.append(f"- {col}: Unable to analyze")
            
            # Numeric summaries with actual values
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                analysis.append(f"\nNUMERIC ANALYSIS:")
                for col in numeric_cols:
                    try:
                        stats = df[col].describe()
                        analysis.append(f"- {col}: Mean={stats['mean']:.2f}, Max={stats['max']}, Min={stats['min']}")
                    except Exception:
                        pass
            
            # Top values for categorical columns
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                analysis.append(f"\nTOP VALUES IN CATEGORICAL COLUMNS:")
                for col in text_cols[:3]:  # Limit to first 3 text columns
                    try:
                        top_values = df[col].value_counts().head(3)
                        analysis.append(f"- {col} top values:")
                        for value, count in top_values.items():
                            analysis.append(f"  '{value}': {count} occurrences")
                    except Exception:
                        pass
            
            return "\n".join(analysis)
            
        except Exception as e:
            logger.error(f"Error in detailed analysis: {str(e)}")
            return f"Basic analysis: {len(df)} records, columns: {', '.join(df.columns)}"
    
    def _format_sample_data(self, df, limit=10):
        """Format sample data in a readable way for AI"""
        try:
            if df.empty:
                return "No data available"
            
            # Get sample rows
            sample_df = df.head(limit)
            
            # Format as clear text with actual values
            formatted_data = []
            formatted_data.append(f"ACTUAL DATA ({len(sample_df)} sample records):")
            formatted_data.append("=" * 50)
            
            for idx, row in sample_df.iterrows():
                formatted_data.append(f"\nRecord {idx + 1}:")
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        formatted_data.append(f"  {col}: (empty)")
                    else:
                        formatted_data.append(f"  {col}: {value}")
            
            return "\n".join(formatted_data)
            
        except Exception as e:
            logger.error(f"Error formatting sample data: {str(e)}")
            return f"Sample data formatting error: {str(e)}"

    def format_ai_response(self, response_text):
        """Format AI response to ensure proper line breaks and structure"""
        try:
            # Clean up the response and add proper line breaks
            formatted_response = response_text
            
            # Add line breaks before section headers
            formatted_response = re.sub(r'(\*\*[^*]+:\*\*)', r'\n\n\1\n', formatted_response)
            
            # Add line breaks before numbered lists
            formatted_response = re.sub(r'(\d+\.)', r'\n\1', formatted_response)
            
            # Add line breaks before bullet points
            formatted_response = re.sub(r'(\s+-\s)', r'\n- ', formatted_response)
            
            # Clean up multiple line breaks
            formatted_response = re.sub(r'\n{3,}', '\n\n', formatted_response)
            
            # Clean up leading/trailing whitespace
            formatted_response = formatted_response.strip()
            
            logger.info(f"FORMATTING_DEBUG: Original length: {len(response_text)}")
            logger.info(f"FORMATTING_DEBUG: Formatted length: {len(formatted_response)}")
            logger.info(f"FORMATTING_DEBUG: First 200 chars: {formatted_response[:200]}")
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
        return response_text  # Return original if formatting fails
    
    def _create_detailed_system_prompt(self, view_data, data_summary, wants_chart=False):
        """Create a detailed system prompt that emphasizes using actual data WITH PROPER FORMATTING"""
        chart_instruction = ""
        if wants_chart:
            chart_instruction = """
        6. If a chart was created, reference it in your response and explain what insights it reveals
        7. If a chart was requested but couldn't be created, explain why and suggest alternatives
        """
        
        return f"""
        You are an expert data analyst working with REAL Tableau data. You must ALWAYS use the actual data values provided, never placeholders.
        
        CRITICAL INSTRUCTIONS:
        1. Use ONLY actual values from the data provided (real numbers, real names, real dates)
        2. NEVER use placeholders like [Customer Name], [Revenue Value], [Top Product], etc.
        3. If you don't have specific data to answer a question, say "The data doesn't contain information about X"
        4. Always cite specific numbers, names, and values from the actual dataset
        5. Provide concrete examples using real data points
        {chart_instruction}
        
        FORMATTING REQUIREMENTS (VERY IMPORTANT):
        - Use double asterisks (**) for section headers that end with colon (:)
        - Put each numbered item on a new line: 1. First item\n2. Second item\n3. Third item
        - Put each bullet point on a new line with dash: \n- First point\n- Second point
        - Add line breaks (\n) between sections
        - Example format:
        **Section Header:**
        
        1. First numbered item
        2. Second numbered item
        3. Third numbered item
        
        **Another Section:**
        
        - First bullet point
        - Second bullet point
        - Third bullet point
        
        DATASET CONTEXT:
        - Report contains: {view_data['record_count']:,} actual records
        - Columns: {', '.join(view_data['columns'])}
        - Last updated: {view_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        
        RESPONSE FORMAT:
        - Start with key findings using actual numbers
        - Use proper line breaks and formatting as shown above
        - Include actionable insights based on the actual data
        - Use numbered lists for items in sequence
        - Use bullet points for recommendations or insights
        - End with concrete recommendations
        
        Remember: Every number, name, and value in your response must come from the actual data provided.
        Use proper line breaks and formatting to make your response easy to read.
        """

# Initialize the chatbot
chatbot = TableauChatBot()

@app.route('/')
def index():
    """Serve the HTML interface"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Tableau Chat Assistant</h1><p>HTML interface not found. Please ensure 'index.html' exists.</p>"

@app.route('/api/connect', methods=['POST'])
def connect_tableau():
    """Initialize Tableau connection and return available views"""
    try:
        success = chatbot.connect_to_tableau()
        if success:
            views = chatbot.get_available_views()
            return jsonify({
                'status': 'success',
                'message': 'Connected to Tableau Server',
                'views': views or []
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to connect to Tableau Server. Please check your credentials and network connection.'
            }), 500
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Connection error: {str(e)}'
        }), 500

@app.route('/api/views', methods=['GET'])
def get_views():
    """Get list of available Tableau views"""
    try:
        views = chatbot.get_available_views()
        if views:
            return jsonify({
                'status': 'success',
                'views': views
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No views available or connection issue'
            }), 404
    except Exception as e:
        logger.error(f"Views error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error fetching views: {str(e)}'
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages and return AI responses with optional charts"""
    try:
        data = request.json
        view_id = data.get('view_id')
        message = data.get('message')
        
        if not view_id or not message:
            return jsonify({
                'status': 'error',
                'message': 'Missing view_id or message'
            }), 400
        
        # Validate view_id exists in filtered views
        if not chatbot.available_views is None:
            try:
                # Get the filtered views
                filtered_views = chatbot.get_available_views()
                view_ids = [view['id'] for view in filtered_views] if filtered_views else []
                
                if view_id not in view_ids and len(view_ids) > 0:
                    return jsonify({
                        'status': 'error',
                        'message': f'Invalid or restricted view_id: {view_id}. Please select from the available views.'
                    }), 400
            except Exception as validation_error:
                logger.warning(f"Could not validate view_id: {str(validation_error)}")
        
        # Get AI response (may include chart data)
        logger.info(f"Incoming chat request - View ID: {view_id}, Message: {message}")
        result = chatbot.analyze_data_with_ai(view_id, message)
        
        # DETAILED LOGGING OF RESULT
        logger.info("=== RESULT ANALYSIS ===")
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        if 'response' in result:
            logger.info(f"Response text length: {len(result['response'])}")
            logger.info(f"Response preview: {result['response'][:100]}...")
        
        if 'chart' in result:
            logger.info("CHART DATA FOUND!")
            chart_data = result['chart']
            logger.info(f"Chart data type: {type(chart_data)}")
            
            if isinstance(chart_data, dict):
                logger.info(f"Chart data keys: {list(chart_data.keys())}")
                
                # Log chart structure
                if 'data' in chart_data:
                    logger.info(f"Chart has 'data' key: {type(chart_data['data'])}")
                    if isinstance(chart_data['data'], list):
                        logger.info(f"Chart data list length: {len(chart_data['data'])}")
                        if len(chart_data['data']) > 0:
                            logger.info(f"First data item keys: {list(chart_data['data'][0].keys()) if isinstance(chart_data['data'][0], dict) else 'Not a dict'}")
                
                if 'layout' in chart_data:
                    logger.info(f"Chart has 'layout' key: {type(chart_data['layout'])}")
                    if isinstance(chart_data['layout'], dict):
                        logger.info(f"Layout keys: {list(chart_data['layout'].keys())}")
                
                # Print first 500 characters of chart data for inspection
                try:
                    chart_json_str = json.dumps(chart_data, indent=2)[:500]
                    logger.info(f"Chart data preview:\n{chart_json_str}...")
                except Exception as json_error:
                    logger.error(f"Could not serialize chart data to JSON: {json_error}")
            else:
                logger.warning(f"Chart data is not a dict: {chart_data}")
        else:
            logger.info("No chart data in result")
        
        # Prepare response
        response_data = {
            'status': 'success',
            'response': result.get('response'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add chart data if available
        if 'chart' in result:
            response_data['chart'] = result['chart']
            response_data['chart_description'] = result.get('chart_description')
            logger.info(f"CHART_DEBUG: Chart data included in response")
            
            # DETAILED RESPONSE DATA LOGGING
            logger.info("=== RESPONSE DATA ANALYSIS ===")
            logger.info(f"Response data keys: {list(response_data.keys())}")
            logger.info(f"Chart included: {'chart' in response_data}")
            logger.info(f"Chart description: {response_data.get('chart_description', 'None')}")
            
            # Log the size of the response
            try:
                response_size = len(json.dumps(response_data))
                logger.info(f"Total response size: {response_size} characters")
            except:
                logger.info("Could not calculate response size")
                
        else:
            logger.info(f"CHART_DEBUG: No chart data in response")
        
        # FINAL RESPONSE LOGGING
        logger.info("=== FINAL RESPONSE ===")
        logger.info(f"Sending response with status: {response_data['status']}")
        logger.info(f"Response contains chart: {'chart' in response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': f'Error processing request: {str(e)}'
        }), 500

@app.route('/api/refresh/<view_id>', methods=['POST'])
def refresh_view_data(view_id):
    """Force refresh data for a specific view"""
    try:
        view_data = chatbot.get_view_data(view_id, force_refresh=True)
        if view_data:
            return jsonify({
                'status': 'success',
                'message': 'Data refreshed successfully',
                'record_count': view_data['record_count'],
                'timestamp': view_data['timestamp'].isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to refresh view data. Please check view permissions and connection.'
            }), 404
    except Exception as e:
        logger.error(f"Refresh error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Refresh error: {str(e)}'
        }), 500

@app.route('/api/view/<view_id>/info', methods=['GET'])
def get_view_info(view_id):
    """Get footer information (Text/Category fields and Dataset Summary) for a specific view"""
    try:
        view_data = chatbot.get_view_data(view_id)
        if view_data:
            df = view_data['data']
            
            # Get view information from available views
            view_info = None
            if chatbot.available_views is not None:
                filtered_views = chatbot.get_available_views()
                for view in filtered_views:
                    if view['id'] == view_id:
                        view_info = view
                        break
            
            footer_info = chatbot._format_column_headers(df, view_info)
            
            return jsonify({
                'status': 'success',
                'footer_info': footer_info,
                'timestamp': view_data['timestamp'].isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to retrieve view information.'
            }), 404
    except Exception as e:
        logger.error(f"View info error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving view info: {str(e)}'
        }), 500

@app.route('/api/view/<view_id>/data', methods=['GET'])
def get_view_data_endpoint(view_id):
    """Get data summary for a specific view"""
    try:
        view_data = chatbot.get_view_data(view_id)
        if view_data:
            # Return metadata and sample data
            df = view_data['data']
            sample_data = df.head(10).to_dict('records') if not df.empty else []
            
            return jsonify({
                'status': 'success',
                'metadata': {
                    'record_count': view_data['record_count'],
                    'columns': view_data['columns'],
                    'timestamp': view_data['timestamp'].isoformat()
                },
                'sample_data': sample_data,
                'summary': chatbot._prepare_data_summary(df)
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to retrieve view data. Please check view permissions and try again.'
            }), 404
    except Exception as e:
        logger.error(f"View data error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving view data: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'tableau_connected': chatbot.tableau_conn is not None,
        'cached_views': len(chatbot.cached_data),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/debug/chart/<view_id>', methods=['POST'])
def debug_chart_generation(view_id):
    """Debug endpoint to test chart generation"""
    try:
        data = request.json
        user_question = data.get('message', 'Show me a bar chart')
        
        logger.info(f"DEBUG: Starting chart generation for view_id: {view_id}")
        logger.info(f"DEBUG: User question: {user_question}")
        
        # Get the data
        view_data = chatbot.get_view_data(view_id)
        if not view_data:
            return jsonify({
                'status': 'error',
                'message': 'No view data available',
                'debug_info': 'view_data is None'
            })
        
        df = view_data['data']
        logger.info(f"DEBUG: DataFrame shape: {df.shape}")
        logger.info(f"DEBUG: DataFrame columns: {list(df.columns)}")
        logger.info(f"DEBUG: DataFrame dtypes: {df.dtypes.to_dict()}")
        
        if df.empty:
            return jsonify({
                'status': 'error',
                'message': 'DataFrame is empty',
                'debug_info': 'DataFrame has no rows'
            })
        
        # Test chart detection
        wants_chart, chart_type = chatbot.detect_chart_intent(user_question, df)
        logger.info(f"DEBUG: Chart detection - wants_chart: {wants_chart}, chart_type: {chart_type}")
        
        if not wants_chart:
            return jsonify({
                'status': 'info',
                'message': 'No chart intent detected',
                'debug_info': f'Question: {user_question}'
            })
        
        # Test chart creation
        chart_data, chart_description = chatbot.create_visualization(df, chart_type, user_question)
        logger.info(f"DEBUG: Chart creation result - chart_data exists: {chart_data is not None}")
        logger.info(f"DEBUG: Chart description: {chart_description}")
        
        if chart_data:
            return jsonify({
                'status': 'success',
                'message': 'Chart generated successfully',
                'chart': chart_data,
                'chart_description': chart_description,
                'debug_info': {
                    'chart_type': chart_type,
                    'data_shape': df.shape,
                    'columns': list(df.columns)
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Chart creation failed: {chart_description}',
                'debug_info': {
                    'chart_type': chart_type,
                    'data_shape': df.shape,
                    'columns': list(df.columns)
                }
            })
            
    except Exception as e:
        logger.error(f"DEBUG: Exception in chart generation: {str(e)}")
        import traceback
        logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': f'Exception: {str(e)}',
            'debug_info': traceback.format_exc()
        })

if __name__ == '__main__':
    # Initialize Tableau connection on startup
    print("🚀 Starting Tableau Chat Assistant...")
    print("📊 Connecting to Tableau Server...")
    
    if chatbot.connect_to_tableau():
        print("✅ Successfully connected to Tableau!")
        views = chatbot.get_available_views()
        print(f"📋 Found {len(views) if views else 0} available reports")
    else:
        print("❌ Failed to connect to Tableau Server")
    
    print("🌐 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)