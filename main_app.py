"""
Bundestag Speech Analysis - Main Application

This module implements the Streamlit interface for the parliamentary speech analysis system.
For a working demo, you'll need to:
1. Download the Open Discourse Dataset
2. Process the data using the scripts in the data_processing directory
3. Generate the FAISS index using creat_faiss_index.py
"""

import streamlit as st
import os
import pandas as pd
import base64
import plotly.express as px
from typing import Optional
from retriever import Retriever
# Must be the first Streamlit command
st.set_page_config(
    page_title="Bundestag Speech Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store')
ASSETS_PATH = os.path.join(BASE_DIR, 'assets')

# Ensure directories exist
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(ASSETS_PATH, exist_ok=True)

# Update image paths to use relative paths
IMG_PATH = os.path.join(ASSETS_PATH, 'image.png')
COVER_PATH = os.path.join(ASSETS_PATH, 'cover.jpg')

# Constants
PARTY_ABBREVIATIONS = {
    "Bündnis 90/Die Grünen": "Grüne",
    "Christlich Demokratische Union Deutschlands/Christlich-Soziale Union in Bayern": "CDU/CSU",
    "Sozialdemokratische Partei Deutschlands": "SPD",
    "Freie Demokratische Partei": "FDP",
    "Alternative für Deutschland": "AfD",
    "DIE LINKE.": "LINKE",
    "fraktionslos": "Fr.los"
}

class RetrieverSingleton:
    _instance: Optional[Retriever] = None
    
    @classmethod
    def get_instance(cls) -> Optional[Retriever]:
        if cls._instance is None:
            try:
                vector_store_path = os.path.join(os.path.dirname(__file__), 'vector_store')
                cls._instance = Retriever(vector_store_path=vector_store_path)
            except Exception as e:
                st.error(f"Error initializing retriever: {str(e)}")
                return None
        return cls._instance

@st.cache_resource(show_spinner=True)
def load_retriever():
    return RetrieverSingleton.get_instance()

@st.cache_data(ttl=3600)
def get_filter_options(_retriever):
    """Cache filter options for better performance"""
    options = {
        'parties': sorted(set(
            "Unknown Party" if p == "-1" else PARTY_ABBREVIATIONS.get(p, p)
            for p in {c['metadata'].get('party','') for c in _retriever.chunk_metadata}
        )),
        'years': sorted(set(
            y for y in {c['metadata'].get('date','')[:4] for c in _retriever.chunk_metadata}
            if y.isdigit() and 2000 <= int(y) <= 2022
        )),
        'topics': sorted(set(
            topic['name'] for c in _retriever.chunk_metadata 
            for topic in c.get('topics', [])
            if isinstance(topic, dict) and 'name' in topic
        )),
        'speech_types': sorted(set(
            c['metadata'].get('speech_type', '') 
            for c in _retriever.chunk_metadata
        ))
    }
    return options

# Also let's add caching for the search results
@st.cache_data(ttl=300)
def cached_search(_retriever, query, top_k, filters):
    """Cache search results for better performance"""
    return _retriever.search(query=query, top_k=top_k, filters=filters)

@st.cache_data(ttl=3600)
def prepare_analytics_data(speech_df):
    """Pre-process data for analytics to improve performance"""
    try:
        # Convert date to datetime with flexible parsing
        speech_df['date'] = pd.to_datetime(
            speech_df['date'],
            format='mixed',  # Allow mixed formats
            yearfirst=True   # Assume year comes first (YYYY-MM-DD)
        )
        
        # Extract year after successful date conversion
        speech_df['year'] = speech_df['date'].dt.year
        
        # Pre-calculate topic frequencies
        topic_counts = {}
        for topics in speech_df['topics']:
            for topic in topics:
                if isinstance(topic, dict):
                    topic_name = topic.get('name', '')
                    if topic_name:
                        topic_counts[topic_name] = topic_counts.get(topic_name, 0) + 1
        
        return speech_df, topic_counts
        
    except Exception as e:
        st.error(f"Error processing dates: {str(e)}")
        # Log some sample dates for debugging
        sample_dates = speech_df['date'].head()
        st.write("Sample dates:", sample_dates.tolist())
        raise e

@st.cache_data(ttl=3600)
def get_topic_trends(filtered_df, selected_topics):
    """Calculate topic trends efficiently"""
    topic_data = []
    
    # Group by year first to reduce iterations
    yearly_groups = filtered_df.groupby('year')
    
    for year, group in yearly_groups:
        for topic in selected_topics:
            # Count speeches that contain this topic
            count = sum(1 for topics in group['topics'] if topic in topics)
            if count > 0:
                topic_data.append({
                    'Year': int(year),
                    'Topic': topic,
                    'Count': count
                })
    
    return pd.DataFrame(topic_data)

@st.cache_data(ttl=3600)
def get_party_topic_distribution(filtered_df, selected_topics):
    """Calculate party-topic distribution efficiently"""
    party_topic_data = []
    
    # Group by party first
    party_groups = filtered_df.groupby('party')
    
    for party, group in party_groups:
        for topic in selected_topics:
            count = sum(1 for topics in group['topics'] if topic in topics)
            if count > 0:
                party_topic_data.append({
                    'Party': party,
                    'Topic': topic,
                    'Count': count
                })
    
    return pd.DataFrame(party_topic_data)

def main():
    # Initialize retriever first
    retriever = load_retriever()
    if not retriever:
        st.error("Error: Could not initialize retriever")
        return

    # Load logo
    if os.path.exists(IMG_PATH):
        with open(IMG_PATH, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode()
    else:
        st.warning("Please add image.png to the assets directory")
    
    col1, col2 = st.columns([1,4])
    with col1:
        if os.path.exists(IMG_PATH):
            with open(IMG_PATH, "rb") as f:
                encoded_img = base64.b64encode(f.read()).decode()
            st.markdown(
                f'<div class="logo-container">'
                f'<img src="data:image/png;base64,{encoded_img}" class="logo-img">'
                f'</div>',
                unsafe_allow_html=True
            )

    with col2:
        st.title("Bundestag Speech Analysis")
        st.markdown("*Explore and analyze parliamentary speeches with advanced semantic search*")

    # Sidebar controls
    with st.sidebar:
        app_mode = st.selectbox("Select Mode", ["Search", "Analytics", "About"])
        if app_mode == "Search":
            top_k = st.slider("Number of Results", 1, 20, 5)

        theme_mode = st.radio("Select Theme", ["Light", "Dark"], index=0)
        st.session_state['theme_mode'] = theme_mode

    # Apply theme
    st.markdown(get_theme_css(theme_mode), unsafe_allow_html=True)

    # Run appropriate mode with retriever
    if app_mode == "Search":
        run_search_mode(retriever, top_k)
    elif app_mode == "Analytics":
        run_analytics_mode(retriever)
    else:
        show_about_page()

def get_theme_css(theme_mode):
    if theme_mode == "Dark":
        return """<style>
            .stApp { background-color: #2B2825; color: #F5F5F5; }
            .css-1d391kg { background-color: #3F3A36; }
            .stTextInput > div > div { background-color: #564F4A; border-bottom: 2px solid #DB7093; }
            .stButton > button { background-color: #990F3D; color: #FFF1E5; }
            h1,h2,h3 { color: #F5F5F5; }
            .logo-container { width:220px; height:220px; margin:2rem auto 1rem auto; }
            .logo-img { width:100%; height:100%; border-radius:50%; object-fit:cover; border:3px solid #990F3D; }
        </style>"""
    else:
        return """<style>
            .stApp { background-color: #FFF7F0; color: #33302E; }
            .css-1d391kg { background-color: #F2E9E1; }
            .stTextInput > div > div { background-color: #F9EDE5; border-bottom: 2px solid #990F3D; }
            .stButton > button { background-color: #990F3D; color: #FFF1E5; }
            h1,h2,h3 { color: #33302E; }
            .logo-container { width:220px; height:220px; margin:2rem auto 1rem auto; }
            .logo-img { width:100%; height:100%; border-radius:50%; object-fit:cover; border:3px solid #990F3D; }
        </style>"""

def run_search_mode(retriever, top_k):
    st.markdown("## Search Mode")
    
    # Get cached filter options
    filter_options = get_filter_options(retriever)
    
    with st.form(key='search_form'):
        query = st.text_input("🔎 Search Query", placeholder="Enter search phrase...")

        # Create columns for main filters
        colA, colB, colC = st.columns(3)
        filters = {}

        with colA:
            sel_party = st.selectbox(
                "🏛️ Party", 
                ["All"] + filter_options['parties']
            )

        with colB:
            years = filter_options['years']
            min_year = min(int(y) for y in years)
            max_year = max(int(y) for y in years)
            
            year_range = st.slider(
                "📅 Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )

        with colC:
            sel_topic = st.selectbox(
                "📑 Topic", 
                ["All"] + filter_options['topics']
            )

        # Advanced filters
        with st.expander("🔍 Advanced Filters"):
            col1, col2 = st.columns(2)
            with col1:
                min_words = st.number_input("Minimum Words", min_value=0, value=0)
            with col2:
                min_confidence = st.slider(
                    "Topic Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5
                )
                max_words = st.number_input("Maximum Words", min_value=0, value=1000)

        search_button = st.form_submit_button("Search")
        
        if search_button and query:
            if sel_party != "All":
                inv_map = {v:k for k,v in PARTY_ABBREVIATIONS.items()}
                filters['party'] = inv_map.get(sel_party, sel_party)
            
            filters['year_range'] = year_range
            
            if sel_topic != "All":
                filters['topic'] = sel_topic

            filters['word_count_range'] = (min_words, max_words)
            filters['min_confidence'] = min_confidence
            
            show_search_results(retriever, query, top_k, filters)

def show_search_results(retriever, query, top_k, filters):
    with st.spinner("Searching..."):
        results = cached_search(retriever, query, top_k, filters)
        
        if results:
            st.success(f"Found {len(results)} relevant result{'s' if len(results) !=1 else ''}.")
            for i, r in enumerate(results, 1):
                with st.expander(f"Result {i}"):
                    md = r.get('metadata', {})
                    party_str = "Unknown Party" if md.get('party') in ["-1", "Unknown Party"] else md.get('party')
                    
                    # Header information
                    st.markdown(f"**Speaker:** {md.get('speaker','Unknown')}")
                    st.markdown(f"**Party:** {party_str}")
                    st.markdown(f"**Date:** {md.get('date','Unknown')}")

                    tabs = st.tabs(["📝 Summary", "📄 Context", "📚 Full Text"])
                    
                    # Summary tab
                    with tabs[0]:
                        summary = r.get('summary', '')
                        if summary:
                            st.markdown("### Generated Summary")
                            st.write(summary)
                            
                            topics = r.get('topics', [])[:3]
                            if topics:
                                st.markdown("### Key Topics")
                                for topic in topics:
                                    if isinstance(topic, dict):
                                        name = topic.get('name', '')
                                        confidence = topic.get('confidence', 0)
                                        if name and confidence:
                                            st.markdown(f"- {name} (confidence: {confidence:.2f})")
                                    else:
                                        st.markdown(f"- {topic}")
                        else:
                            st.warning("No generated summary available.")
                    
                    # Context tab
                    with tabs[1]:
                        context = r.get('context', '')
                        if context:
                            st.write(context)
                        else:
                            st.write("No context available.")
                    
                    # Full Text tab
                    with tabs[2]:
                        clean_tab, orig_tab = st.tabs(["Cleaned Text", "Original Text"])
                        with clean_tab:
                            st.write(r.get('cleaned_text', ''))
                        with orig_tab:
                            st.write(md.get('original_text', ''))
        else:
            st.warning("No results found for the given query.")

def run_analytics_mode(retriever):
    st.markdown("## Analytics Mode")
    
    # Get speech data
    speech_data = []
    for chunk in retriever.chunk_metadata:
        metadata = chunk.get('metadata', {})
        topics = chunk.get('topics', [])
        if isinstance(topics, list):
            topics = [t.get('name', t) if isinstance(t, dict) else t for t in topics]
        
        speech_data.append({
            'date': metadata.get('date', ''),
            'party': metadata.get('party', 'Unknown'),
            'speaker': metadata.get('speaker', 'Unknown'),
            'topics': set(topics),
            'total_chunks': 1
        })
    
    if speech_data:
        speech_df = pd.DataFrame(speech_data)
        speech_df, topic_counts = prepare_analytics_data(speech_df)
        
        tabs = st.tabs(["General Analytics", "Topic Analysis", "Top Topics Trends", "Export Data"])
        
        with tabs[0]:
            st.markdown("### Overview Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Speeches", len(speech_df))
            with col2:
                st.metric("Unique Speakers", len(speech_df['speaker'].unique()))
            with col3:
                st.metric("Total Parties", len(speech_df['party'].unique()))
            
            # Party Distribution
            st.markdown("### Party Distribution")
            party_counts = speech_df['party'].value_counts()
            fig = px.pie(
                values=party_counts.values,
                names=party_counts.index,
                title="Distribution by Party"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top Speakers
            st.markdown("### Top 10 Speakers")
            top_speakers = speech_df['speaker'].value_counts().head(10)
            fig = px.bar(
                x=top_speakers.index,
                y=top_speakers.values,
                title="Most Active Speakers"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Timeline of Speeches
            st.markdown("### Timeline of Speeches")
            timeline_data = speech_df.groupby(speech_df['date'].dt.to_period('Y')).size()
            fig = px.line(
                x=timeline_data.index.astype(str),
                y=timeline_data.values,
                title="Number of Speeches by Year",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Number of Speeches"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:  # Topic Analysis
            st.markdown("### Topic Distribution")
            
            # Filter controls
            col1, col2, col3 = st.columns(3)
            with col1:
                start_year = st.selectbox(
                    "Start Year",
                    options=sorted(speech_df['year'].unique()),
                    index=0
                )
            with col2:
                end_year = st.selectbox(
                    "End Year",
                    options=sorted(speech_df['year'].unique()),
                    index=len(speech_df['year'].unique())-1
                )
            with col3:
                selected_party = st.selectbox(
                    "Select Party",
                    options=["All"] + sorted(speech_df['party'].unique())
                )
            
            # Filter data
            filtered_data = speech_df[
                (speech_df['year'] >= start_year) & 
                (speech_df['year'] <= end_year)
            ]
            if selected_party != "All":
                filtered_data = filtered_data[filtered_data['party'] == selected_party]
            
            # Calculate topic distribution
            all_topics = []
            for topics_set in filtered_data['topics']:
                all_topics.extend(topics_set)
            
            topic_dist = pd.Series(all_topics).value_counts().head(10)
            
            # Show topic distribution
            fig = px.bar(
                x=topic_dist.index,
                y=topic_dist.values,
                title=f"Top 10 Topics ({start_year}-{end_year})" + 
                      (f" - {selected_party}" if selected_party != "All" else "")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top speakers for selected topics
            st.markdown("### Top Speakers by Topic")
            selected_topic = st.selectbox(
                "Select Topic to see top speakers",
                options=topic_dist.index
            )
            
            speaker_stats = filtered_data[
                filtered_data['topics'].apply(lambda x: selected_topic in x)
            ]['speaker'].value_counts().head(5)
            
            st.markdown(f"**Top 5 Speakers discussing '{selected_topic}':**")
            for speaker, count in speaker_stats.items():
                st.markdown(f"- {speaker}: {count} speeches")
            
            # Add Party Comparison Section
            st.markdown("### Party Topic Comparison")
            
            # Date range selector for party comparison
            st.markdown("#### Select Date Range for Comparison")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                comp_start_date = pd.to_datetime(
                    st.date_input(
                        "Comparison Start Date",
                        pd.to_datetime(speech_df['date'].min()).date(),
                        key="comp_start_date"
                    )
                ).date()
            with comp_col2:
                comp_end_date = pd.to_datetime(
                    st.date_input(
                        "Comparison End Date",
                        pd.to_datetime(speech_df['date'].max()).date(),
                        key="comp_end_date"
                    )
                ).date()
            
            # Party selector for comparison
            parties_to_compare = st.multiselect(
                "Select Parties to Compare",
                options=sorted(speech_df['party'].unique()),
                default=sorted(speech_df['party'].unique())[:2]  # Default select first two parties
            )
            
            # Topic selector for comparison
            topics_to_compare = st.multiselect(
                "Select Topics to Compare",
                options=sorted(set(t for topics in speech_df['topics'] for t in topics)),
                default=sorted(set(t for topics in speech_df['topics'] for t in topics))[:3]  # Default select first three topics
            )
            
            if parties_to_compare and topics_to_compare:
                # Filter data for comparison
                comp_mask = (
                    (speech_df['date'].dt.date >= comp_start_date) &
                    (speech_df['date'].dt.date <= comp_end_date) &
                    (speech_df['party'].isin(parties_to_compare))
                )
                comparison_df = speech_df[comp_mask]
                
                if not comparison_df.empty:
                    # Calculate topic frequencies for each party
                    party_topic_data = []
                    
                    for party in parties_to_compare:
                        party_speeches = comparison_df[comparison_df['party'] == party]
                        for topic in topics_to_compare:
                            count = sum(1 for topics in party_speeches['topics'] if topic in topics)
                            if count > 0:
                                party_topic_data.append({
                                    'Party': party,
                                    'Topic': topic,
                                    'Count': count
                                })
                    
                    if party_topic_data:
                        comparison_df = pd.DataFrame(party_topic_data)
                        
                        # Create comparison visualizations
                        
                        # 1. Bar chart comparing topics across parties
                        fig1 = px.bar(
                            comparison_df,
                            x='Party',
                            y='Count',
                            color='Topic',
                            title=f"Topic Distribution by Party ({comp_start_date} to {comp_end_date})",
                            barmode='group'
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # 2. Heatmap of topic frequency by party
                        pivot_df = comparison_df.pivot(
                            index='Party',
                            columns='Topic',
                            values='Count'
                        ).fillna(0)
                        
                        fig2 = px.imshow(
                            pivot_df,
                            title="Topic Frequency Heatmap",
                            labels=dict(x="Topic", y="Party", color="Count"),
                            aspect="auto",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # 3. Show percentage distribution
                        st.markdown("#### Percentage Distribution of Topics by Party")
                        
                        # Calculate percentages
                        party_totals = comparison_df.groupby('Party')['Count'].sum()
                        percentage_data = []
                        
                        for party in parties_to_compare:
                            party_data = comparison_df[comparison_df['Party'] == party]
                            total = party_totals[party]
                            for _, row in party_data.iterrows():
                                percentage_data.append({
                                    'Party': party,
                                    'Topic': row['Topic'],
                                    'Percentage': (row['Count'] / total * 100)
                                })
                        
                        # Create percentage distribution chart
                        fig3 = px.bar(
                            pd.DataFrame(percentage_data),
                            x='Party',
                            y='Percentage',
                            color='Topic',
                            title="Topic Distribution Percentage by Party",
                            barmode='stack'
                        )
                        fig3.update_layout(yaxis_title="Percentage (%)")
                        st.plotly_chart(fig3, use_container_width=True)
                        
                    else:
                        st.warning("No data available for the selected combination of parties and topics.")
                else:
                    st.warning("No data available for the selected date range.")
            else:
                st.info("Please select at least one party and one topic to compare.")
        
        with tabs[2]:  # Top Topics Trends
            st.markdown("### Top 3 Topics Over Time")
            
            # Create filters
            col1, col2 = st.columns(2)
            with col1:
                start_date = pd.to_datetime(
                    st.date_input(
                        "Start Date",
                        pd.to_datetime(speech_df['date'].min()).date()
                    )
                ).date()
            with col2:
                end_date = pd.to_datetime(
                    st.date_input(
                        "End Date",
                        pd.to_datetime(speech_df['date'].max()).date()
                    )
                ).date()
            
            # Party multiselect with cached options
            party_options = sorted(speech_df['party'].unique())
            selected_parties = st.multiselect(
                "Select Parties",
                options=party_options,
                default=party_options
            )
            
            # Filter data efficiently
            mask = (
                (speech_df['date'].dt.date >= start_date) &
                (speech_df['date'].dt.date <= end_date) &
                (speech_df['party'].isin(selected_parties))
            )
            filtered_df = speech_df[mask]
            
            if not filtered_df.empty:
                # Get top topics from the filtered data
                all_topics = []
                for topics_set in filtered_df['topics']:
                    all_topics.extend(topics_set)
                
                topic_counts = pd.Series(all_topics).value_counts()
                top_3_topics = topic_counts.head(3).index.tolist()
                
                # Get topic trends
                yearly_topic_counts = get_topic_trends(filtered_df, top_3_topics)
                
                if not yearly_topic_counts.empty:
                    # Create line chart
                    fig = px.line(
                        yearly_topic_counts,
                        x='Year',
                        y='Count',
                        color='Topic',
                        markers=True,
                        title="Top 3 Topics Trend Over Time"
                    )
                    
                    fig.update_layout(
                        xaxis=dict(
                            tickmode='linear',
                            dtick=1,
                            tickformat='d'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display statistics efficiently
                    stats_cols = st.columns(3)
                    for i, topic in enumerate(top_3_topics):
                        with stats_cols[i]:
                            topic_stats = yearly_topic_counts[yearly_topic_counts['Topic'] == topic]
                            total_mentions = topic_stats['Count'].sum()
                            peak_year = topic_stats.loc[topic_stats['Count'].idxmax(), 'Year']
                            
                            st.metric(
                                f"Topic {i+1}: {topic}",
                                f"{total_mentions} mentions",
                                f"Peak year: {peak_year}"
                            )
                else:
                    st.warning("No topic trends found for the selected filters.")
            else:
                st.warning("No data available for the selected date range.")
        
        with tabs[3]:  # Export Data
            st.markdown("### Export Analysis Data")
            
            if not filtered_df.empty:
                # Allow selecting date range for export
                st.markdown("#### Select Date Range for Export")
                col1, col2 = st.columns(2)
                with col1:
                    export_start_date = st.date_input(
                        "Export Start Date",
                        start_date,
                        key="export_start_date"
                    )
                with col2:
                    export_end_date = st.date_input(
                        "Export End Date",
                        end_date,
                        key="export_end_date"
                    )
                
                # Apply export date filter
                export_mask = (
                    (filtered_df['date'].dt.date >= export_start_date) & 
                    (filtered_df['date'].dt.date <= export_end_date)
                )
                export_df = filtered_df[export_mask].copy()
                
                if not export_df.empty:
                    # Format data for export
                    export_df['topics'] = export_df['topics'].apply(lambda x: ', '.join(sorted(x)) if x else '')
                    
                    # Prepare download data
                    csv = export_df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv,
                        file_name=f"bundestag_speeches_{export_start_date}_{export_end_date}.csv",
                        mime="text/csv"
                    )
                    
                    # Show preview
                    st.markdown("### Data Preview")
                    st.dataframe(export_df.head(), use_container_width=True)
                else:
                    st.warning("No data available for the selected export date range")
    else:
        st.error("No speech data available for analysis.")

def show_about_page():
    # Create two columns - one for project info, one for personal info
    main_col, sidebar_col = st.columns([2, 1])
    
    with sidebar_col:
        # Personal Info Card
        st.markdown("""
        <style>
        .personal-card {
            border: 2px solid #e6e6e6;
            border-radius: 10px;
            padding: 0;  /* Removed padding to eliminate top space */
            margin: 10px;
            background-color: #ffffff;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            overflow: hidden;  /* This ensures no gaps */
        }
        .profile-img {
            width: 100%;
            display: block;  /* This removes any image spacing */
            margin: 0;  /* Remove any margin */
            border-radius: 10px 10px 0 0;  /* Round only top corners */
        }
        .content-padding {
            padding: 20px;  /* Add padding to content area only */
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="personal-card">', unsafe_allow_html=True)
            
            # Add your profile picture
            profile_img_path = COVER_PATH  # Use the path from constants
            if os.path.exists(profile_img_path):
                st.image(profile_img_path, use_container_width=True)
            else:
                st.warning("Please add cover.jpg to the assets directory")
            
            # Wrap content in a padded div
            st.markdown('<div class="content-padding">', unsafe_allow_html=True)
            st.markdown("### Fehmi Katar")
            
            st.markdown('<p class="section-header">Education</p>', unsafe_allow_html=True)
            st.markdown("""
            - Actuarial Science (Marmara University)
            - Social Work (Alice Salomon Hochschule)
            - Advanced Data Analytics and Machine Learning:
              - 42 Berlin
              - Code Academy Berlin
              - Google Data Analytics Certification
            """)
            
            st.markdown('<p class="section-header">Expertise</p>', unsafe_allow_html=True)
            st.markdown("""
            - Data Analysis and Machine Learning
            - Natural Language Processing (NLP)
            - Unstructured Data Transformation
            """)
            
            st.markdown('<p class="section-header">Technical Skills</p>', unsafe_allow_html=True)
            skills = ["Python", "SQL", "Tableau", "spaCy", "Hugging Face", "FAISS", "RAG Pipelines"]
            skills_html = " ".join([f'<span class="skill-tag">{skill}</span>' for skill in skills])
            st.markdown(f'<div>{skills_html}</div>', unsafe_allow_html=True)
            
            st.markdown('<p class="section-header">Languages</p>', unsafe_allow_html=True)
            st.markdown("""
            - German (C2)
            - English (C1)
            - Turkish (Native)
            - Kurdish (Native)
            """)
            
            st.markdown('<p class="section-header">Contact</p>', unsafe_allow_html=True)
            st.markdown("""
            - 📧 [katar.fhm@gmail.com](mailto:katar.fhm@gmail.com)
            - [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fehmi-dataanalyst)
            - [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Rojhak)
            """)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close content padding div
            st.markdown('</div>', unsafe_allow_html=True)  # Close personal card div
    
    with main_col:
        st.markdown("## About the Project")
        
        st.markdown("""
        The **Bundestag Speech Analysis Project** is a **Retrieval-Augmented Generation (RAG)**-powered platform 
        designed to analyze German parliamentary speeches from 2000–2022, developed as my final project during a 
        6-month Data Science bootcamp at Code Academy Berlin. It transforms vast amounts of 
        **unstructured textual data** into structured insights through advanced natural language processing (NLP) techniques.
        The system retrieves speech segments based on user queries, providing summaries, topic information, and access 
        to the original text.
        
        **Data Source**: This project uses the [Open Discourse Dataset](https://dataverse.harvard.edu/dataverse/opendiscourse), 
        a comprehensive corpus of plenary proceedings from the German Parliament (Bundestag).
        """)
        
        # Key Capabilities Section
        st.markdown("### Key Capabilities")
        
        capabilities = {
            "Data Transformation": [
                "Turning unstructured text into structured, actionable data",
                "Extraction of speech metadata (topics, dates, speakers, party affiliations)"
            ],
            "Semantic Search with RAG": [
                "FAISS (Facebook AI Similarity Search) for efficient vector search",
                "SentenceTransformer models for generating embeddings",
                "Context-aware retrieval of relevant speech segments"
            ],
            "Summarization and Topic Extraction": [
                "spaCy for extractive summarization and topic identification",
                "Hierarchical taxonomy for topic classification",
                "Detection of key terms and subtopic relationships"
            ]
        }
        
        for cap, details in capabilities.items():
            st.markdown(f"**{cap}:**")
            for detail in details:
                st.markdown(f"- {detail}")
        
        # Real-World Applications
        st.markdown("### Real-World Applications")
        
        st.markdown("""
        This project showcases how **RAG and NLP** can transform unstructured data into meaningful insights:
        - **Policy Research and Advocacy**: Analyze political discourse, identify key themes, and track trends
        - **Corporate Knowledge Management**: Enable efficient information retrieval from vast repositories
        - **Customer Support**: Develop intelligent Q&A systems
        - **Education and Research**: Assist in summarizing and contextualizing large datasets
        """)
        
        # Techniques and Tools
        st.markdown("### Techniques and Tools")
        
        st.markdown("""
        The project combines advanced data processing and NLP tools:
        - **spaCy**: For extractive summarization and topic identification
        - **FAISS**: Vector-based search engine for efficient retrieval
        - **SentenceTransformers**: For generating semantic embeddings
        - **Streamlit**: Intuitive user interface
        - **Python Data Pipeline**: Seamless data processing and storage
        """)
        
        # Conclusion
        st.markdown("### Conclusion")
        
        st.markdown("""
        The **Bundestag Speech Analysis Project** demonstrates the transformative potential of 
        **RAG and NLP** in deriving actionable insights from unstructured data. Whether for 
        legislative analysis, enterprise knowledge management, or customer support, this platform 
        shows how modern AI can bridge the gap between raw data and meaningful insights.
        """)
        
        # After the initial project description, add:
        
        st.markdown("### Technical Architecture")
        
        st.markdown("""
        The project follows a multi-layer architecture for processing and analyzing parliamentary speeches:
        
        **1. Data Preprocessing Pipeline**
        - CSV files cleaning and structuring
        - Metadata extraction (speakers, parties, dates)
        - JSON conversion and organization
        - Data validation and quality checks
        
        **2. Database Layer**
        - Processed JSON files storage
        - FAISS vector database for embeddings
        - Metadata storage and indexing
        
        **3. Processing Layer**
        - Text cleaning with spaCy
        - Topic extraction and classification
        - Summary generation
        - Vector embeddings creation using SentenceTransformers
        
        **4. Search & Analytics Layer**
        - Semantic search functionality
        - Topic trend analysis
        - Party comparison tools
        - Timeline visualizations
        
        **5. User Interface Layer**
        - Interactive search interface
        - Analytics dashboard
        - Dynamic visualizations
        - Project documentation
        """)
        
        # Add system diagram
        st.markdown("### System Flow")
        st.markdown("""
        ```mermaid
        graph TD
            A[Raw Data] --> B[Data Preprocessing]
            B --> C[Database Layer]
            C --> D[Processing Layer]
            D --> E[Search & Analytics]
            E --> F[User Interface]
            
            subgraph Preprocessing
            B1[CSV Files] --> B2[Clean & Structure]
            B2 --> B3[JSON Conversion]
            end
            
            subgraph Core System
            C1[FAISS Store] --> D1[Text Processing]
            D1 --> D2[Topic Extraction]
            D2 --> E1[Semantic Search]
            E1 --> E2[Analytics]
            end
        ```
        """)
        
        st.markdown("### Key Files and Components")
        
        st.markdown("""
        The system consists of several key components:
        
        **Data Processing:**
        - **new_enhanced_cleaning.py**: Initial data preprocessing and cleaning of raw parliamentary data
        - **process_cleaned_files.py**: Secondary processing and structuring of cleaned data
        
        **Core RAG Components:**
        - **retrieval.py**: Core component that handles:
          - Topic-based search functionality
          - Context retrieval for speeches
          - Initial summary generation
        - **creat_faiss_index.py**: Creates and manages vector database for semantic search
        - **creat_summary.py**: Advanced text summarization and topic extraction
        
        **Application Layer:**
        - **main_app.py**: User interface and core application functionality
        - **analytics.py**: Data analysis and visualization components
        
        Each component is designed to be modular and maintainable, with the retrieval system acting as the bridge 
        between the data storage and user interface layers. This architecture allows for efficient search and 
        retrieval of relevant speech segments based on both semantic similarity and topic relevance.
        """)

if __name__=="__main__":
    main()
