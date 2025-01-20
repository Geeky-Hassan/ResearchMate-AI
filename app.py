# conda activate "D:\Python_Projects\Generative AI projects\ML Project\mllab"

# import streamlit as st
# import pickle
# from sentence_transformers import SentenceTransformer, util
# import torch
# from pathlib import Path

# # Load the sentences and embeddings
# @st.cache_resource
# def load_data():
#     ROOT = Path.cwd()  # Root folder of the project
#     MODEL = ROOT / "models"  # Folder inside the root directory

#     # List of file names to load
#     file_names = ["Titles.pkl", "URLS.pkl", "Embedding_Titles.pkl", "Embedding_URLS.pkl"]

#     # Dictionary to store loaded data
#     loaded_files = {}

#     # Load all files
#     for file_name in file_names:
#         file_path = MODEL / file_name  # Construct full file path using pathlib
#         print(f"Loading file: {file_path}")
#         try:
#             # Open the pickle file and load the data
#             with open(file_path, "rb") as f:
#                 data = pickle.load(f)
#             print(f"File '{file_name}' loaded successfully from: {file_path}")
            
#             # Store the loaded data in the dictionary
#             loaded_files[file_name] = data
#         except Exception as e:
#             print(f"Error loading file '{file_name}': {e}")
    
#     # Extract the loaded data as separate variables
#     titles = loaded_files.get("Titles.pkl", None)
#     urls = loaded_files.get("URLS.pkl", None)
#     embeddings_titles = loaded_files.get("Embedding_Titles.pkl", None)
#     embeddings_urls = loaded_files.get("Embedding_URLS.pkl", None)

#     # Return the four variables
#     return titles, urls, embeddings_titles, embeddings_urls

# # Call the function and unpack the returned values into variables
# titles, urls, embeddings_titles, embeddings_urls = load_data()


# @st.cache_resource
# def load_model():
#     try:
#         # Load the sentence transformer model
#         model = SentenceTransformer('all-MiniLM-L6-v2')
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# model = load_model()


# st.title("Rsearch Paper Recommender")

# if titles is not None and urls is not None and embeddings_titles is not None and embeddings_urls is not None and model is not None:
#     user_query = st.text_input("Enter your topic of interest here:")

#     if user_query:
#         # Generate embeddings for the user query
#         query_embedding = model.encode(user_query)

#         # Calculate cosine similarities
#         cosine_scores_titles = util.cos_sim(embeddings_titles, query_embedding)
#         cosine_scores_urls = util.cos_sim(embeddings_urls, query_embedding)

#         # Find top k most similar
#         top_similar_papers_titles = torch.topk(cosine_scores_titles, dim=0, k=5, sorted=True)
#         top_similar_papers_urls = torch.topk(cosine_scores_urls, dim=0, k=5, sorted=True)
        
#         # Reshape the top_similar_papers.indices to handle issues
#         top_title_indices = top_similar_papers_titles.indices.squeeze().tolist()
#         top_url_indices = top_similar_papers_titles.indices.squeeze().tolist()

#         st.subheader("Recommended Papers:")
#         index = 1
#         for i in top_similar_papers_titles.indices:
#             for i in top_similar_papers_urls.indices:
#                 st.write(f"{index}- Title: {titles[i.item()]}")
#                 st.write(f"URL: {urls[i.item()]}")
#                 st.write("")
#                 index += 1
# else:
#     st.error("Could not load the models, data.")














import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer, util
import torch
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="ResearchMate AI",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .project-title {
        font-size: 4rem !important;
        text-align: center;
        padding: 2rem 0;
        color: #1e3d59;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-weight: bold;
        background: linear-gradient(120deg, #1e3d59, #17c3b2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .project-subtitle {
        color: #666;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .team-member {
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
    .search-container {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with Team Information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100)  # You can replace this with your own logo
    st.markdown("### 👥 Team Members")
    st.markdown("""
        <div style='background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <div class='team-member'>
                <span style='color: #1e3d59; font-weight: bold;'>👨‍💻 Noor Ul Hassan</span>
            </div>
            <div class='team-member'>
                <span style='color: #1e3d59; font-weight: bold;'>👨‍💻 Umair Tahir</span>
            </div>
            <div class='team-member'>
                <span style='color: #1e3d59; font-weight: bold;'>👨‍💻 Zohaib</span>
            </div>
            <div class='team-member' style='border-bottom: none;'>
                <span style='color: #1e3d59; font-weight: bold;'>👨‍💻 Afnan</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📝 Project Info")
    st.info("""
        This project is part of the Machine Learning A2 Lab Final Project. 
        
        It uses advanced NLP techniques to recommend relevant research papers based on user queries.
    """)

# Main Content
# Centered Project Title with Book Emoji
st.markdown('<h1 class="project-title">📚 ResearchMate AI</h1>', unsafe_allow_html=True)
st.markdown("""
    <p class="project-subtitle">
    An Intelligent Research Paper Recommendation System
    </p>
""", unsafe_allow_html=True)

# Load the sentences and embeddings
@st.cache_resource
def load_data():
    ROOT = Path.cwd()
    MODEL = ROOT / "models"
    
    file_names = ["Titles.pkl", "URLS.pkl", "Embedding_Titles.pkl", "Embedding_URLS.pkl"]
    loaded_files = {}
    
    for file_name in file_names:
        file_path = MODEL / file_name
        print(f"Loading file: {file_path}")
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            print(f"File '{file_name}' loaded successfully from: {file_path}")
            loaded_files[file_name] = data
        except Exception as e:
            print(f"Error loading file '{file_name}': {e}")
    
    return (loaded_files.get("Titles.pkl", None),
            loaded_files.get("URLS.pkl", None),
            loaded_files.get("Embedding_Titles.pkl", None),
            loaded_files.get("Embedding_URLS.pkl", None))

# Load model
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load data and model
titles, urls, embeddings_titles, embeddings_urls = load_data()
model = load_model()

# Main search interface
if titles is not None and urls is not None and embeddings_titles is not None and embeddings_urls is not None and model is not None:
    # Search container
    user_query = st.text_input(
        "🔍 Enter your research topic of interest:",
        placeholder="e.g., machine learning, artificial intelligence, deep learning",
        key="search_box"
    )

    if user_query:
        with st.spinner("🔄 Finding relevant papers..."):
            query_embedding = model.encode(user_query)
            cosine_scores_titles = util.cos_sim(embeddings_titles, query_embedding)
            cosine_scores_urls = util.cos_sim(embeddings_urls, query_embedding)
            
            top_similar_papers_titles = torch.topk(cosine_scores_titles, dim=0, k=5, sorted=True)
            top_similar_papers_urls = torch.topk(cosine_scores_urls, dim=0, k=5, sorted=True)

            st.markdown('<h2 style="color: #1e3d59; text-align: center; margin: 2rem 0;">📚 Recommended Papers</h2>', unsafe_allow_html=True)
            
            for idx, (i, j) in enumerate(zip(top_similar_papers_titles.indices, 
                                           top_similar_papers_urls.indices), 1):
                with st.container():
                    st.markdown(f"""
                    <div style="
                        background-color: white;
                        padding: 1.5rem;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        margin: 1rem 0;
                        border-left: 5px solid #17c3b2;
                    ">
                        <h3 style="color: #1e3d59; margin-bottom: 0.5rem;">Paper {idx}</h3>
                        <p style="color: #34495e; font-size: 1.1rem; margin-bottom: 0.5rem;">
                            <strong>Title:</strong> {titles[i.item()]}
                        </p>
                        <p style="color: #576574;">
                            <strong>URL:</strong> <a href="{urls[j.item()]}" target="_blank" 
                            style="color: #17c3b2;">{urls[j.item()]}</a>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
else:
    st.error("""
        ⚠️ System Configuration Error
        
        Could not load the required models and data. Please check:
        - Model files are present in the correct directory
        - All required packages are installed
        - System has sufficient memory
        
        Contact the development team for support.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p style="font-size: 1.2rem;">Made with ❤️ by Team ResearchMate AI</p>
        <p>© 2025 All rights reserved</p>
    </div>
""", unsafe_allow_html=True)