# BundesRAG-Analytics

🎯 A Retrieval-Augmented Generation (RAG) powered platform for semantic analysis of German parliamentary speeches, combining advanced NLP with interactive visualization.

## Overview
Transform parliamentary discourse into actionable insights using cutting-edge AI. This project leverages FAISS vector search, SentenceTransformers, and spaCy to enable intelligent exploration of Bundestag speeches from 2000-2022.

### Key Features
- 🔍 **Semantic Search**: Advanced context-aware search beyond keywords
- 📊 **Real-time Analytics**: Dynamic visualization of political trends
- 🎯 **Topic Tracking**: AI-powered topic extraction and analysis
- 🔄 **RAG Pipeline**: Combines retrieval-based and generative approaches
- 🌐 **Multilingual Support**: Handles both German and English interfaces

### Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-3776AB?style=for-the-badge)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge)

## Project Overview
This project implements a comprehensive analysis system for German parliamentary speeches, featuring:
- **Semantic Search**: Advanced search functionality using FAISS and SentenceTransformers
- **Topic Analysis**: Automated topic extraction and trend visualization
- **Interactive Dashboard**: Real-time analytics and data exploration
- **Context-Aware Retrieval**: Smart context extraction around search terms
- **Multi-Language Support**: Handles both German and English interfaces

### Key Features
- **Smart Search**: Uses semantic similarity to find relevant speeches beyond keyword matching
- **Topic Tracking**: Analyzes how topics evolve over time and across political parties
- **Party Analysis**: Compare speaking patterns and topic focus between parties
- **Speaker Insights**: Track individual speaker contributions and themes
- **Interactive Visualizations**: Dynamic charts and graphs for data exploration

### Technical Implementation
- **RAG Pipeline**: Combines retrieval-based and generative approaches
- **Vector Search**: FAISS for efficient similarity search
- **NLP Processing**: spaCy and Transformers for text analysis
- **Interactive UI**: Streamlit for the web interface
- **Data Processing**: Efficient batch processing of parliamentary data

## Project Structure
```
rag_model/
├── data_processing/
│   ├── new_enhanced_cleaning.py    # Initial CSV cleaning
│   └── process_cleaned_files.py    # JSON conversion
├── assets/                         # Images and static files
├── vector_store/                   # FAISS index storage
├── main_app.py                     # Main Streamlit application
├── retriever.py                    # Core retrieval functionality
├── creat_summary.py               # Text summarization
├── creat_faiss_index.py           # Vector database management
└── requirements.txt               # Project dependencies
```

## Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/yourusername/bundestag-speech-analysis.git
cd bundestag-speech-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
```

3. Project Setup:
   - Create necessary directories:
   ```bash
   mkdir -p rag_model/assets rag_model/vector_store
   ```
   - The repository includes required images in:
     - `rag_model/assets/image.png` (logo)
     - `rag_model/assets/cover.jpg` (profile image)

4. Data Processing:
   a. Download the [Open Discourse Dataset](https://dataverse.harvard.edu/dataverse/opendiscourse)
   b. Process the data:
   ```bash
   python data_processing/new_enhanced_cleaning.py --input path/to/your/raw_data --output path/to/cleaned_data
   python data_processing/process_cleaned_files.py --input path/to/cleaned_data --output path/to/processed_data
   ```

5. Run the application:
```bash
streamlit run main_app.py
```

## Directory Structure
```
bundestag-speech-analysis/
├── rag_model/
│   ├── assets/              # Create this directory
│   │   ├── image.png       # Add your logo
│   │   └── cover.jpg       # Add your profile image
│   ├── vector_store/       # Will be created during processing
│   └── [other files...]
├── data/                   # Create this for your dataset
│   ├── raw/               # Place downloaded data here
│   ├── cleaned/           # Cleaned data output
│   └── processed/         # Final processed data
└── [other files...]
```

## Configuration
Before running the application, you'll need to:
1. Update paths in your configuration to match your setup
2. Ensure all required directories exist
3. Place necessary assets in the correct locations

## Data Source
This project uses the [Open Discourse Dataset](https://dataverse.harvard.edu/dataverse/opendiscourse)

## License
MIT License - See LICENSE file for details
