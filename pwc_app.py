import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from datetime import datetime
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# Define all functions at the top
def clean_currency(x):
    if isinstance(x, str):
        # Remove 'R' prefix, commas, and spaces
        return float(x.replace('R', '').replace(',', '').replace(' ', ''))
    return x

def calculate_pct_changes(df, value_col):
    # Sort by date to ensure correct calculation of changes
    df = df.sort_values('date')
    # Calculate percentage change
    pct_change = df[value_col].pct_change() * 100
    return pct_change

def extract_bank_info(file_path):
    try:
        # Read the first few lines manually to get the header information
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Extract date and bank name from the first two lines
        date = lines[0].strip().split(',')[1].strip()
        bank_name = lines[1].strip().split(',')[1].strip()
        
        # Read the file for processing the credit impairments
        # Skip the first 6 rows as they contain header information
        df = pd.read_csv(file_path, skiprows=6, dtype=str)  # Read all columns as string initially
        
        # Find the row where Item Number is 194
        impairment_row = df[df['Item Number'] == '194']
        
        if not impairment_row.empty:
            try:
                # Get the value from column 7 (0-based index 6)
                impairment_value = impairment_row.iloc[0, 6]
                # Convert to float and multiply by 1000 (as values are in thousands)
                impairment_value = float(impairment_value.replace(',', '')) * 1000 if pd.notnull(impairment_value) else None
            except (ValueError, AttributeError) as e:
                print(f"Error converting impairment value in {file_path}: {e}")
                impairment_value = None
        else:
            print(f"No item number 194 found in {os.path.basename(file_path)}")
            impairment_value = None
        
        return {
            'date': date,
            'bank_name': bank_name,
            'credit_impairments': impairment_value,
            'file_name': os.path.basename(file_path)
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def process_all_files():
    # Get all CSV files in the pwc_data directory
    csv_files = glob.glob('pwc_data/*.csv')
    
    # Process each file and collect results
    results = []
    for file_path in csv_files:
        info = extract_bank_info(file_path)
        if info is not None:
            results.append(info)
    
    # Convert results to DataFrame for easy viewing/export
    if results:
        results_df = pd.DataFrame(results)
        # Format the credit_impairments as currency
        results_df['credit_impairments'] = results_df['credit_impairments'].apply(
            lambda x: f"R{x:,.2f}" if pd.notnull(x) else None
        )
        # Sort by bank name
        results_df = results_df.sort_values('bank_name')
        return results_df
    else:
        return pd.DataFrame(columns=['date', 'bank_name', 'credit_impairments', 'file_name'])


st.title("PWC Bank Impairments")

if st.sidebar.button("Process new data..."):
    results_df = process_all_files()
    results_df.to_csv('bank_impairments_summary.csv', index=False)

if st.sidebar.button("Retrieve data..."):
    results_df = pd.read_csv("bank_impairments_summary.csv")
    with st.expander("View bank_impairments_summary..."):
        st.dataframe(results_df)

# Load the data
results_df = pd.read_csv("bank_impairments_summary.csv")
economic_indicators = pd.read_csv("economic_indicators.csv")
# st.write("Original data shape:", results_df.shape)
# st.write("Original data and types:")
# st.write(results_df.head())
# st.write(results_df.dtypes)

# After loading the data, let's examine the economic indicators structure
# st.write("Economic Indicators Columns:", economic_indicators.columns.tolist())

# Assuming the date column might have a different name, let's check the first few rows
# st.write("Economic Indicators Head:", economic_indicators.head())

# After loading the economic indicators data
# Reshape economic indicators from wide to long format
economic_indicators_long = economic_indicators.melt(
    id_vars=['Indicator'],
    var_name='date',
    value_name='value'
)

# Convert date strings to datetime (new format)
economic_indicators_long['date'] = pd.to_datetime(economic_indicators_long['date'], format='%Y/%m/%d')

# Create a formatted date string that matches the bank data format
economic_indicators_long['date_formatted'] = economic_indicators_long['date'].dt.strftime('%B %Y')

# Pivot the indicators to get them as columns
economic_indicators_processed = economic_indicators_long.pivot(
    index='date',
    columns='Indicator',
    values='value'
).reset_index()

# Rename columns to remove spaces and special characters
economic_indicators_processed.columns = [
    'date' if col == 'date' else col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '') 
    for col in economic_indicators_processed.columns
]

# Calculate percentage changes for each economic indicator
economic_changes = economic_indicators_processed.copy()
numeric_columns = economic_indicators_processed.columns.drop('date')

for col in numeric_columns:
    economic_changes[f'pct_change_{col}'] = calculate_pct_changes(economic_changes, col)

# Convert credit_impairments to numeric
results_df['credit_impairments'] = results_df['credit_impairments'].apply(clean_currency)


# Convert dates and ensure proper sorting
results_df['date'] = pd.to_datetime(results_df['date'], format='%B %Y')
# Sort the dataframe by date
results_df = results_df.sort_values('date', ascending=True)
# Create formatted date after sorting
results_df['date_formatted'] = results_df['date'].dt.strftime('%B %Y')

# After loading and processing the data, before creating the charts
# Calculate total impairments for each date
total_by_date = results_df.groupby('date_formatted')['credit_impairments'].sum().reset_index()
total_by_date['bank_name'] = 'Total All Banks'

# Add total to the main dataframe
results_df_with_total = pd.concat([results_df, total_by_date[['date_formatted', 'bank_name', 'credit_impairments']]])

# Prepare impairments changes
impairments_changes = []
for bank in results_df['bank_name'].unique():
    bank_data = results_df[results_df['bank_name'] == bank].copy()
    bank_data['pct_change_impairments'] = calculate_pct_changes(bank_data, 'credit_impairments')
    impairments_changes.append(bank_data)

impairments_changes_df = pd.concat(impairments_changes)

# Merge the datasets
modeling_df = pd.merge(
    impairments_changes_df[['date', 'bank_name', 'pct_change_impairments']],
    economic_changes[[
        'date',
        *[f'pct_change_{col}' for col in numeric_columns]
    ]],
    on='date',
    how='left'
)

# Display raw data
with st.expander("Raw Data"):
    st.dataframe(
        results_df[['date_formatted', 'bank_name', 'credit_impairments']]
        .sort_values(['date_formatted', 'bank_name'])
    )
    
with st.expander("Economic Data"):
    st.dataframe(economic_indicators)

# Display the changes
st.header("Changes Analysis for Modeling")

# Display impairments changes
with st.expander("View Impairments Changes"):
    st.write("Percentage Changes in Credit Impairments by Bank")
    pivot_impairments = pd.pivot_table(
        modeling_df,
        values='pct_change_impairments',
        index='date',
        columns='bank_name'
    ).round(2)
    st.dataframe(pivot_impairments)

# Display economic indicator changes
with st.expander("View Economic Indicator Changes"):
    st.write("Percentage Changes in Economic Indicators")
    economic_changes_display = economic_changes[[
        'date',
        *[f'pct_change_{col}' for col in numeric_columns]
    ]].round(2)
    st.dataframe(economic_changes_display)

# Create a correlation matrix
with st.expander("View Correlation Matrix"):
    st.write("Correlation between Impairments Changes and Economic Indicator Changes")
    
    # Prepare correlation data
    correlation_data = modeling_df.drop(['date', 'bank_name'], axis=1).corr()
    
    # Create correlation heatmap
    fig_corr = px.imshow(
        correlation_data,
        title="Correlation Matrix of Changes",
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    
    # Update layout for better readability
    fig_corr.update_layout(
        xaxis_tickangle=-45,
        width=800,
        height=800
    )
    
    st.plotly_chart(fig_corr)

# Display the final modeling dataset
with st.expander("View Complete Modeling Dataset"):
    st.write("Final Dataset for Modeling (Percentage Changes)")
    st.dataframe(modeling_df.round(2))

# Visualization section
st.header("Credit Impairments Analysis")

viz_type = st.selectbox(
    "Select Visualization Type",
    ["Line Plot", "Bar Plot"]
)

if viz_type == "Line Plot":
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add individual bank traces to primary y-axis
    for bank in results_df['bank_name'].unique():
        bank_data = results_df[results_df['bank_name'] == bank]
        fig.add_trace(
            go.Scatter(
                x=bank_data['date_formatted'],
                y=bank_data['credit_impairments'],
                name=bank,
                mode='lines'
            )
        )
    
    # Add total trace to secondary y-axis
    total_data = total_by_date
    fig.add_trace(
        go.Scatter(
            x=total_data['date_formatted'],
            y=total_data['credit_impairments'],
            name='Total All Banks',
            mode='lines',
            line=dict(width=3, dash='dash'),
            yaxis='y2'
        )
    )
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title='Credit Impairments Over Time by Bank',
        xaxis_title="Date",
        yaxis_title="Credit Impairments (ZAR)",
        yaxis2=dict(
            title="Total Credit Impairments (ZAR)",
            overlaying='y',
            side='right',
            tickformat=','
        ),
        yaxis=dict(
            tickformat=',',
            title='Credit Impairments (ZAR)'
        ),
        hovermode='x unified',
        xaxis_tickangle=45,
        legend_title="Bank Name",
        # Adjust legend position to prevent overlap with secondary y-axis
        legend=dict(x=1.1, y=1)
    )
else:
    fig = px.bar(
        results_df_with_total,
        x='date_formatted',
        y='credit_impairments',
        color='bank_name',
        title='Credit Impairments Over Time by Bank',
        barmode='group',
        labels={
            'date_formatted': 'Date',
            'credit_impairments': 'Credit Impairments (ZAR)',
            'bank_name': 'Bank Name'
        },
        category_orders={'date_formatted': sorted(results_df_with_total['date_formatted'].unique(), 
                                                key=lambda x: pd.to_datetime(x, format='%B %Y'))}
    )

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Create normalized data for comparison
st.header("Normalized Credit Impairments Analysis (Base = 100)")

# Create a copy of the dataframe for normalization
normalized_df = results_df_with_total.copy()

# Calculate normalized values for each bank and total
for entity in normalized_df['bank_name'].unique():
    entity_data = normalized_df[normalized_df['bank_name'] == entity]
    first_value = entity_data['credit_impairments'].iloc[0]
    normalized_df.loc[normalized_df['bank_name'] == entity, 'normalized_impairments'] = \
        (normalized_df.loc[normalized_df['bank_name'] == entity, 'credit_impairments'] / first_value) * 100

# Update the normalized chart
if viz_type == "Line Plot":
    fig_normalized = px.line(
        normalized_df,
        x='date_formatted',
        y='normalized_impairments',
        color='bank_name',
        title='Normalized Credit Impairments Over Time by Bank (Base = 100)',
        labels={
            'date_formatted': 'Date',
            'normalized_impairments': 'Normalized Credit Impairments (Base = 100)',
            'bank_name': 'Bank Name'
        },
        category_orders={'date_formatted': sorted(normalized_df['date_formatted'].unique(), 
                                                key=lambda x: pd.to_datetime(x, format='%B %Y'))}
    )
    
    # Make the total line thicker and dashed
    for trace in fig_normalized.data:
        if trace.name == 'Total All Banks':
            trace.line.width = 3
            trace.line.dash = 'dash'
else:
    fig_normalized = px.bar(
        normalized_df,
        x='date_formatted',
        y='normalized_impairments',
        color='bank_name',
        title='Normalized Credit Impairments Over Time by Bank (Base = 100)',
        barmode='group',
        labels={
            'date_formatted': 'Date',
            'normalized_impairments': 'Normalized Credit Impairments (Base = 100)',
            'bank_name': 'Bank Name'
        },
        category_orders={'date_formatted': sorted(normalized_df['date_formatted'].unique(), 
                                                key=lambda x: pd.to_datetime(x, format='%B %Y'))}
    )

# Customize normalized layout
fig_normalized.update_layout(
    xaxis_title="Date",
    yaxis_title="Normalized Credit Impairments (Base = 100)",
    legend_title="Bank Name",
    hovermode='x unified',
    xaxis_tickangle=45,
    # Add a horizontal line at y=100 to show the baseline
    shapes=[dict(
        type='line',
        yref='y',
        y0=100,
        y1=100,
        xref='paper',
        x0=0,
        x1=1,
        line=dict(
            color='gray',
            width=1,
            dash='dash'
        )
    )]
)

# Display the normalized plot
st.plotly_chart(fig_normalized, use_container_width=True)

# Display summary statistics with formatted values
with st.expander("Summary Statistics (in ZAR)"):
    summary_stats = results_df.groupby('bank_name')['credit_impairments'].agg([
        ('Mean', 'mean'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Total', 'sum')
    ]).round(2)

    # Format summary statistics to include commas for readability
    for col in summary_stats.columns:
        summary_stats[col] = summary_stats[col].apply(lambda x: "R{:,.2f}".format(x))

    st.write(summary_stats)

# Before the regression analysis section, add this filtering step
st.header("Regression Analysis by Bank")

# First, identify banks with sufficient data points
valid_banks = []
for bank in results_df['bank_name'].unique():
    bank_data = modeling_df[modeling_df['bank_name'] == bank].copy()
    
    # Prepare X (independent variables) and y (dependent variable)
    X = bank_data[[f'pct_change_{col}' for col in numeric_columns]]
    y = bank_data['pct_change_impairments']
    
    # Remove rows with inf or nan values
    valid_mask = ~(X.isin([np.inf, -np.inf]).any(axis=1) | X.isna().any(axis=1) | y.isna())
    valid_data_points = sum(valid_mask)
    
    if valid_data_points >= 3:
        valid_banks.append(bank)
    else:
        st.warning(f"Excluding {bank} from analysis - only {valid_data_points} valid data points available")

# Now perform regression only for banks with sufficient data
for bank in valid_banks:
    with st.expander(f"Regression Analysis for {bank}"):
        # Filter data for this bank
        bank_data = modeling_df[modeling_df['bank_name'] == bank].copy()
        
        # Prepare X (independent variables) and y (dependent variable)
        X = bank_data[[f'pct_change_{col}' for col in numeric_columns]]
        y = bank_data['pct_change_impairments']
        
        # Remove rows with inf or nan values
        valid_mask = ~(X.isin([np.inf, -np.inf]).any(axis=1) | X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Add constant for regression
        X = sm.add_constant(X)
        
        try:
            # Run regression
            model = sm.OLS(y, X).fit()
            
            # Display regression summary
            st.write("### Regression Summary")
            st.text(model.summary().as_text())
            
            # Calculate VIF
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            
            st.write("### Variance Inflation Factors (VIF)")
            st.dataframe(vif_data.round(2))
            
            # Create actual vs predicted plot
            fig_pred = go.Figure()
            
            # Add scatter plot of actual vs predicted
            fig_pred.add_trace(go.Scatter(
                x=y,
                y=model.predict(),
                mode='markers',
                name='Actual vs Predicted',
                marker=dict(color='blue')
            ))
            
            # Add 45-degree line
            min_val = min(min(y), min(model.predict()))
            max_val = max(max(y), max(model.predict()))
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='45-degree line',
                line=dict(color='red', dash='dash')
            ))
            
            fig_pred.update_layout(
                title=f'Actual vs Predicted Values for {bank}',
                xaxis_title='Actual Change in Impairments (%)',
                yaxis_title='Predicted Change in Impairments (%)',
                showlegend=True
            )
            
            st.plotly_chart(fig_pred)
            
            # Create residual plot
            fig_resid = go.Figure()
            
            # Add residual scatter plot
            fig_resid.add_trace(go.Scatter(
                x=model.predict(),
                y=model.resid,
                mode='markers',
                name='Residuals',
                marker=dict(color='blue')
            ))
            
            # Add horizontal line at y=0
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            
            fig_resid.update_layout(
                title=f'Residual Plot for {bank}',
                xaxis_title='Predicted Values',
                yaxis_title='Residuals',
                showlegend=True
            )
            
            st.plotly_chart(fig_resid)
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R-squared", f"{model.rsquared:.3f}")
            with col2:
                st.metric("Adjusted R-squared", f"{model.rsquared_adj:.3f}")
            with col3:
                st.metric("F-statistic p-value", f"{model.f_pvalue:.3f}")
            
            # Display significant coefficients
            st.write("### Significant Coefficients (p < 0.05)")
            significant_coef = pd.DataFrame({
                'Variable': model.pvalues.index,
                'Coefficient': model.params,
                'P-value': model.pvalues
            })
            significant_coef = significant_coef[significant_coef['P-value'] < 0.05]
            st.dataframe(significant_coef.round(4))
            
        except Exception as e:
            st.error(f"Error running regression for {bank}: {str(e)}")

# Add a section for total impairments regression
with st.expander("Regression Analysis for Total Banking Sector"):
    # Calculate total impairments changes
    total_impairments = results_df.groupby('date')['credit_impairments'].sum().reset_index()
    total_impairments['pct_change_total'] = calculate_pct_changes(total_impairments, 'credit_impairments')
    
    # Merge with economic indicators
    total_regression_data = pd.merge(
        total_impairments[['date', 'pct_change_total']],
        economic_changes[[
            'date',
            *[f'pct_change_{col}' for col in numeric_columns]
        ]],
        on='date',
        how='inner'
    )
    
    # Remove rows with inf or nan values
    X_total = total_regression_data[[f'pct_change_{col}' for col in numeric_columns]]
    y_total = total_regression_data['pct_change_total']
    
    valid_mask = ~(X_total.isin([np.inf, -np.inf]).any(axis=1) | X_total.isna().any(axis=1) | y_total.isna())
    X_total = X_total[valid_mask]
    y_total = y_total[valid_mask]
    
    if len(X_total) >= len(numeric_columns) + 1:
        # Add constant
        X_total = sm.add_constant(X_total)
        
        try:
            # Run regression
            model_total = sm.OLS(y_total, X_total).fit()
            
            # Display regression summary
            st.write("### Total Banking Sector Regression Summary")
            st.text(model_total.summary().as_text())
            
            # Display VIF for total sector
            vif_data_total = pd.DataFrame()
            vif_data_total["Variable"] = X_total.columns
            vif_data_total["VIF"] = [variance_inflation_factor(X_total.values, i) for i in range(X_total.shape[1])]
            
            st.write("### Variance Inflation Factors (VIF) - Total Sector")
            st.dataframe(vif_data_total.round(2))
            
        except Exception as e:
            st.error(f"Error running regression for total sector: {str(e)}")
    else:
        st.warning("Insufficient data points for total sector regression after removing missing values")



        

