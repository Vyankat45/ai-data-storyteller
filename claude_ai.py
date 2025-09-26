import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI-Powered Data Storyteller",
    page_icon="📊",
    layout="wide"
)

class DataStorytellerAI:
    def __init__(self):
        self.df = None
        self.insights = []
        self.visualizations = []
    
    def validate_dataset(self, df):
        """Validate uploaded dataset"""
        if df is None or df.empty:
            return False, "Dataset is empty"
        if len(df.columns) < 2:
            return False, "Dataset must have at least 2 columns"
        return True, "Dataset validated successfully"
    
    def perform_eda(self, df):
        """Perform Exploratory Data Analysis"""
        eda_results = {}
        
        # Basic info
        eda_results['shape'] = df.shape
        eda_results['columns'] = list(df.columns)
        eda_results['dtypes'] = df.dtypes.to_dict()
        eda_results['missing_values'] = df.isnull().sum().to_dict()
        
        # Summary statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            eda_results['summary_stats'] = df[numeric_cols].describe().to_dict()
        
        # Correlations
        if len(numeric_cols) > 1:
            eda_results['correlations'] = df[numeric_cols].corr().to_dict()
        
        # Value counts for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        eda_results['value_counts'] = {}
        for col in categorical_cols:
            if df[col].nunique() <= 20:  # Only for columns with reasonable number of unique values
                eda_results['value_counts'][col] = df[col].value_counts().head(10).to_dict()
        
        return eda_results
    
    def generate_insights(self, eda_results, df):
        """Generate natural language insights"""
        insights = []
        
        # Dataset overview
        insights.append(f"📋 **Dataset Overview**: The dataset contains {eda_results['shape'][0]:,} rows and {eda_results['shape'][1]} columns.")
        
        # Missing values analysis
        missing_cols = {k: v for k, v in eda_results['missing_values'].items() if v > 0}
        if missing_cols:
            worst_missing = max(missing_cols.items(), key=lambda x: x[1])
            insights.append(f"⚠️ **Data Quality**: {len(missing_cols)} columns have missing values. '{worst_missing[0]}' has the most missing values ({worst_missing[1]:,} missing).")
        else:
            insights.append("✅ **Data Quality**: No missing values detected in the dataset.")
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"🔢 **Numeric Features**: Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}.")
            
            # Find interesting correlations
            if 'correlations' in eda_results and len(numeric_cols) > 1:
                corr_df = pd.DataFrame(eda_results['correlations'])
                # Find strongest positive correlation (excluding diagonal)
                mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
                corr_vals = corr_df.where(mask)
                max_corr_idx = corr_vals.abs().stack().idxmax()
                max_corr_val = corr_vals.loc[max_corr_idx]
                if abs(max_corr_val) > 0.5:
                    insights.append(f"🔗 **Strong Correlation**: '{max_corr_idx[0]}' and '{max_corr_idx[1]}' show {'positive' if max_corr_val > 0 else 'negative'} correlation ({max_corr_val:.2f}).")
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            insights.append(f"📝 **Categorical Features**: Found {len(categorical_cols)} categorical columns.")
            
            # Find columns with high cardinality
            high_cardinality = [col for col in categorical_cols if df[col].nunique() > df.shape[0] * 0.8]
            if high_cardinality:
                insights.append(f"🔍 **High Cardinality**: Columns {', '.join(high_cardinality)} might be unique identifiers.")
        
        # Value distribution insights
        if 'value_counts' in eda_results:
            for col, counts in eda_results['value_counts'].items():
                if counts:
                    most_common = max(counts.items(), key=lambda x: x[1])
                    total_records = sum(counts.values())
                    percentage = (most_common[1] / total_records) * 100
                    insights.append(f"📊 **'{col}' Distribution**: Most common value is '{most_common[0]}' ({percentage:.1f}% of records).")
        
        return insights
    
    def create_visualizations(self, df):
        """Create meaningful visualizations"""
        visualizations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # 1. Correlation Heatmap (if multiple numeric columns)
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Heatmap of Numeric Features')
            visualizations.append(('Correlation Heatmap', fig))
            plt.close()
        
        # 2. Distribution plot for first numeric column
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            fig, ax = plt.subplots(figsize=(10, 6))
            df[col].hist(bins=30, alpha=0.7, ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            visualizations.append((f'{col} Distribution', fig))
            plt.close()
        
        # 3. Bar chart for first categorical column
        if len(categorical_cols) > 0:
            col = categorical_cols[0]
            if df[col].nunique() <= 15:  # Only if reasonable number of categories
                fig, ax = plt.subplots(figsize=(12, 6))
                value_counts = df[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Top Values in {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                visualizations.append((f'{col} Value Counts', fig))
                plt.close()
        
        # 4. Scatter plot if we have at least 2 numeric columns
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[col1], df[col2], alpha=0.6)
            ax.set_title(f'{col1} vs {col2}')
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            visualizations.append((f'{col1} vs {col2}', fig))
            plt.close()
        
        # 5. Box plot for numeric data if we have categories
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            num_col = numeric_cols[0]
            cat_col = categorical_cols[0]
            if df[cat_col].nunique() <= 10:  # Only if reasonable number of categories
                fig, ax = plt.subplots(figsize=(12, 6))
                df.boxplot(column=num_col, by=cat_col, ax=ax)
                ax.set_title(f'{num_col} by {cat_col}')
                plt.suptitle('')  # Remove default title
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                visualizations.append((f'{num_col} by {cat_col}', fig))
                plt.close()
        
        return visualizations[:3]  # Return at most 3 visualizations as required
    
    def generate_pdf_report(self, insights, visualizations, filename="data_analysis_report.pdf"):
        """Generate PDF report"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        
        # Title
        pdf.cell(0, 10, "Data Analysis Executive Summary", ln=True, align="C")
        pdf.ln(10)
        
        # Date
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        
        # Key Insights Section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Key Insights", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", "", 11)
        for i, insight in enumerate(insights[:6], 1):  # Limit to 6 insights
            # Clean insight text for PDF
            clean_insight = insight.replace('📋', '').replace('⚠️', '').replace('✅', '').replace('🔢', '').replace('🔗', '').replace('📝', '').replace('🔍', '').replace('📊', '').strip()
            clean_insight = clean_insight.replace('**', '')
            
            pdf.cell(0, 8, f"{i}. {clean_insight}", ln=True)
            pdf.ln(2)
        
        # Visualizations section
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Data Visualizations", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, f"Generated {len(visualizations)} key visualizations showing data patterns,", ln=True)
        pdf.cell(0, 8, "correlations, and distributions. See dashboard for interactive charts.", ln=True)
        
        # Recommendations
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Recommendations", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, "1. Monitor data quality and address missing values where identified", ln=True)
        pdf.cell(0, 8, "2. Investigate strong correlations for potential business insights", ln=True)
        pdf.cell(0, 8, "3. Use categorical distributions for targeted analysis", ln=True)
        pdf.cell(0, 8, "4. Consider outlier detection for numeric variables", ln=True)
        
        return pdf.output(dest="S").encode("latin-1")

def main():
    st.title("🤖 AI-Powered Data Storyteller")
    st.markdown("*Automated data analysis, insights generation, and storytelling dashboard*")
    
    # Initialize the AI assistant
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = DataStorytellerAI()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a section:", 
                               ["📤 Data Upload", "📊 Analysis & Insights", "📈 Visualizations", "📄 Generate Report"])
    
    if page == "📤 Data Upload":
        st.header("Dataset Upload & Validation")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.ai_assistant.df = df
                
                # Validate dataset
                is_valid, message = st.session_state.ai_assistant.validate_dataset(df)
                
                if is_valid:
                    st.success(message)
                    st.write("### Dataset Preview")
                    st.dataframe(df.head())
                    
                    st.write("### Dataset Info")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", df.shape[0])
                    with col2:
                        st.metric("Columns", df.shape[1])
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    
                    st.write("### Column Types")
                    st.dataframe(pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.values,
                        'Non-Null Count': df.count().values,
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    }))
                else:
                    st.error(message)
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    elif page == "📊 Analysis & Insights":
        st.header("Automated Data Analysis & Insights")
        
        if st.session_state.ai_assistant.df is not None:
            df = st.session_state.ai_assistant.df
            
            if st.button("🔍 Generate AI Insights"):
                with st.spinner("Analyzing data and generating insights..."):
                    # Perform EDA
                    eda_results = st.session_state.ai_assistant.perform_eda(df)
                    
                    # Generate insights
                    insights = st.session_state.ai_assistant.generate_insights(eda_results, df)
                    st.session_state.ai_assistant.insights = insights
                    
                    st.success("Analysis complete!")
            
            # Display insights if available
            if st.session_state.ai_assistant.insights:
                st.write("### 🧠 AI-Generated Insights")
                for insight in st.session_state.ai_assistant.insights:
                    st.markdown(f"- {insight}")
                
                # Show detailed statistics
                with st.expander("📈 Detailed Statistics"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Numeric Columns Summary**")
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            st.dataframe(df[numeric_cols].describe())
                    
                    with col2:
                        st.write("**Missing Values Analysis**")
                        missing_df = pd.DataFrame({
                            'Column': df.columns,
                            'Missing Count': df.isnull().sum().values,
                            'Missing %': (df.isnull().sum() / len(df) * 100).values
                        })
                        st.dataframe(missing_df[missing_df['Missing Count'] > 0])
        else:
            st.warning("Please upload a dataset first!")
    
    elif page == "📈 Visualizations":
        st.header("Data Visualizations")
        
        if st.session_state.ai_assistant.df is not None:
            df = st.session_state.ai_assistant.df
            
            if st.button("📊 Generate Visualizations"):
                with st.spinner("Creating visualizations..."):
                    visualizations = st.session_state.ai_assistant.create_visualizations(df)
                    st.session_state.ai_assistant.visualizations = visualizations
                    st.success(f"Generated {len(visualizations)} visualizations!")
            
            # Display visualizations if available
            if st.session_state.ai_assistant.visualizations:
                for title, fig in st.session_state.ai_assistant.visualizations:
                    st.write(f"### {title}")
                    st.pyplot(fig)
                    st.divider()
        else:
            st.warning("Please upload a dataset first!")
    
    elif page == "📄 Generate Report":
        st.header("Executive Summary Report")
        
        if (st.session_state.ai_assistant.df is not None and 
            st.session_state.ai_assistant.insights and 
            st.session_state.ai_assistant.visualizations):
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Generate PDF Report"):
                    with st.spinner("Generating PDF report..."):
                        pdf_bytes = st.session_state.ai_assistant.generate_pdf_report(
                            st.session_state.ai_assistant.insights,
                            st.session_state.ai_assistant.visualizations
                        )
                        
                        st.success("PDF report generated successfully!")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
            
            with col2:
                st.info("**Report will include:**\n"
                       "- Executive summary\n"
                       "- Key insights\n"
                       "- Data quality assessment\n"
                       "- Recommendations")
            
            # Preview insights
            st.write("### Report Preview - Key Insights")
            for insight in st.session_state.ai_assistant.insights[:5]:
                st.markdown(f"• {insight}")
                
        else:
            st.warning("Please complete data upload, analysis, and visualization steps first!")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**AI-Powered Data Storyteller**")
    st.sidebar.markdown("Built with Streamlit, Pandas, and AI")

if __name__ == "__main__":
    main()