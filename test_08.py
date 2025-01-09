import streamlit as st
import pandas as pd
from datetime import date
from PIL import Image
import requests
import io
import time
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
 
 
# Custom styling
custom_html = """
    <style>
        .banner {
            width: 100%;
            height: 100px;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #FFD700;
        }
    </style>
    <div class="banner">
        <center><img src="https://logotyp.us/file/super-kings.svg" width="200" height="300"></center>
    </div>
"""
st.markdown(custom_html, unsafe_allow_html=True)
 
 
# Sidebar for CSV upload
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
 
# Helper Functions
 
def get_csv_text(files):
    for csv_file in files:
        if csv_file is not None:
            try:
                # Attempt to read the file
                df = pd.read_csv(csv_file)
                if df.empty:
                    st.error("The uploaded CSV file is empty. Please upload a valid file.")
                    return ""
                return df.to_string(index=False)
            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file has no data. Please check the file and try again.")
            except Exception as e:
                st.error(f"An error occurred while reading the CSV file: {e}")
    return ""
 
 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)
 
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    print(f"Generated embeddings: {embeddings}")
 
 
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, respond with "answer is not available in the context".
   
    Context:
    {context}
   
    Question:
    {question}
 
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1.0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)
 
def process_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]
 
def fetch_image_with_retry(url, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
                return response.content
            else:
                raise Exception(f"Invalid image response: {response.status_code}")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                st.error(f"Failed to load image after {retries} attempts: {e}")
                return None
 
def get_drive_view_url_and_direct_link(url):
    """Converts Google Drive URLs into direct download and view links."""
    if "drive.google.com" in url:
        # Extract the file ID from the shareable URL
        file_id = url.split("/")[-2]
        view_link = f"https://drive.google.com/file/d/{file_id}/view"
        direct_link = f"https://drive.google.com/uc?id={file_id}&export=download"
        return view_link, direct_link
    return url, url
 
# Main Application
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write("### File Preview:")
    st.sidebar.dataframe(df)
 
    # Tabs for functionalities
    tab2, tab1 = st.tabs(["Free Text search", "Chat with GeminiAI"])
 
    with tab1:
        st.header("Chat with CSV using Gemini ")
 
        user_question = st.text_input("Ask a Question from the CSV Files")
 
        if user_question:
            with st.spinner("Processing..."):
                response = process_user_input(user_question)
                st.write("Reply: ", response)
 
                # Extract image URLs from the DataFrame (assuming URLs are in the "URL" column)
                if "URL" in df.columns:
                    image_urls = df["URL"].dropna().tolist()
 
                    # Display images in a grid layout
                    st.write("### Related Images")
                    cols = st.columns(3)  # Display 3 images per row
 
                    for i, url in enumerate(image_urls):
                        with cols[i % 3]:  # Use modulus to create rows of 3
                            if "drive.google.com" in url:
                                view_link, direct_link = get_drive_view_url_and_direct_link(url)
                                if direct_link:
                                    image_content = fetch_image_with_retry(direct_link)
                                    if image_content:
                                        image = Image.open(io.BytesIO(image_content))
                                        st.image(image, use_container_width=True)
                                        if view_link:
                                            st.markdown(f'<a href="{view_link}" target="_blank" style="color: blue;">Open in Google Drive</a>', unsafe_allow_html=True)
                            else:
                                st.image(url, use_container_width=True)
                                st.markdown(f'<a href="{url}" target="_blank" style="color: blue;">{url}</a>', unsafe_allow_html=True)
 
 
    with tab2:
        st.header("Free Text search")
        # Initialize session state
        if "selected_player" not in st.session_state:
            st.session_state.selected_player = None
        if "players" not in st.session_state:
            st.session_state.players = []
        if "display_index" not in st.session_state:
            st.session_state.display_index = 0
       
        available_players = ["Mitchell Santner", "Nishant Sindhu", "Moeen Ali", "Ajay Mandal", "Ben Stokes",
            "Ajinkya Rahane", "Shivam Dube", "Deepak Chahar", "Devon Conway", "Maheesh Theekshana",
            "R Russell", "Akash Singh", "Gregory King", "Lakshmi", "Tushar Deshpande", "Ms Dhoni",
            "Suresh Raina", "Ruturaj Gaikwad", "Simarjeet Singh", "Ravindra Jadeja", "Eric Simon",
            "Shaik Rasheed", "Stephen Fleming", "Subhranshu Senapati", "Dwayne Bravo", "Ambati Rayudu",
            "Bhagath Varma", "Tommy Simsek", "Sanjay Natarajan", "Prashant Solanki", "Rajvardhan Hangargekar",
            "Dwaine Pretorius", "Matheesha Pathirana", "Mukesh Choudhary", "Kasi", "Gerald Coetzee",
            "David Miller", "Faf Du Plessis", "Lahiru Milantha", "Imran Tahir", "Saiteja Mukkamalla",
            "Rusty Theron", "Cameron Stevenson", "Zia Shahzad", "Cody Chetty", "Milind Kumar",
            "Sami Aslam", "Calvin Savage", "Muhammad Mohsin", "Zia Ul Haq"]
         # Function to add a player
        def add_player():
            new_player = st.session_state.new_player
            if new_player and new_player not in st.session_state.players:
                st.session_state.players.append(new_player)
       
        # Function to remove a player
        def remove_player(player):
            if player in st.session_state.players:
                st.session_state.players.remove(player)
       
        # Input area for adding players via dropdown
        st.write("##### Select a player name")
        col1 ,col2, col3 = st.columns([3,1,1])
        with col1:
            st.selectbox(
                " ",
                options=[""] + available_players,
                key="new_player",
                on_change=add_player,
                label_visibility="collapsed",
            )
       
        # Display added players as tags in a single line
        st.write("##### Selected Players")
        if st.session_state.players:
            for player in st.session_state.players:
                # Create a button with "✖️" to remove player
                if st.button(f"✖️ {player}", key=f"remove_{player}"):
                    remove_player(player)
        else:
            st.info("No players added yet. Please add players to proceed.")
       
        # Input for location
        st.write("")  # Spacer
        col5, col6, col8, col9, col10= st.columns(5)
        with col5:
            Action = st.selectbox(
                "ACTION",
                ["All", "Discussion", "Celebration", "Batting", "Award","poses","interview","posing", "celebrating"],
                label_visibility="visible",
            )
        with col6:
        # Combining Day/Night, Environment, and Distance in a single Location filter dropdown
            activity = st.selectbox(
                "ACTIVITY",
                ["Unspecified", "Day","Night","Outdoor","Indoor","Unknown","Close"],
                label_visibility="visible"
            )
       
        with col8:
            start_date = st.date_input(
                "From Date",
                date.today(),
                label_visibility="visible",
            )
        with col9:
            end_date = st.date_input(
                "To Date",
                date.today(),
                label_visibility="visible",
            )
 
        with col10:
            no_of_faces = st.number_input(
                "NO OF FACES",
                min_value=0,
                value=0,  # Default value
                step=1,
                label_visibility="visible",
            )
       
        # Function to filter dataframe based on action
        def filter_by_Action(df, action="all"):
            if "Action" in df.columns:
                if action != "all":
                    # Check if the Caption contains the selected action keyword (case-insensitive)
                    return df[df["Action"].str.contains(action, case=False, na=False)]
                else:
                    # If no action is provided, return the dataframe unfiltered
                    return df
            else:
                st.warning("The uploaded CSV does not contain a 'action' column. Please include it.")
                return df
       
        # Function to filter dataframe based on place
        def filter_by_no_of_faces(df, no_of_faces=0):
            if "No_of_faces" in df.columns:
                return df[df["No_of_faces"] == no_of_faces]
            else:
                st.warning("The uploaded CSV does not contain a 'No_of_faces' column. Please include it.")
                return df
               
 
        # Function to filter dataframe based on location
        def filter_by_activity(df, day_night=None, environment=None, distance=None):
            if day_night and "Day_Night" in df.columns:
                df = df[df["Day_Night"].str.contains(day_night, case=False, na=False)]
 
            if environment and "Environment" in df.columns:
                df = df[df["Environment"].str.contains(environment, case=False, na=False)]
 
            if distance and "Distance" in df.columns:
                df = df[df["Distance"].str.contains(distance, case=False, na=False)]
 
            return df
       
        def filter_by_date(df, from_date, to_date):
            if "Date" in df.columns:
                # Convert the 'Date' column to pandas datetime with infer_datetime_format for flexible formats
                df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors="coerce")
               
                # Convert Streamlit date inputs to pandas datetime
                from_date = pd.to_datetime(from_date.strftime("%Y-%m-%d"))
                to_date = pd.to_datetime(to_date.strftime("%Y-%m-%d"))
               
                # Filter the DataFrame for dates within the specified range
                return df[(df["Date"] >= from_date) & (df["Date"] <= to_date)]
            else:
                st.warning("The uploaded CSV does not contain a 'Date' column. Please include it.")
                return df
               
        # Function to filter players with the same URL
        def filter_by_same_url(df):
            # Group by URL and filter groups where all players in the group have the same URL
            grouped = df.groupby("URL").nunique()
            valid_urls = grouped[grouped > 1].index  # URLs with more than one unique name
            return df[df["URL"].isin(valid_urls)]
       
        # Functionality for the yellow button
        # Functionality for the yellow button
        if st.button("Generate Image", key="yellow_button"):
            st.session_state.display_index = 0  # Reset index for new generation
 
            if st.session_state.players and uploaded_file:
                # Filter by selected player names
                filtered_df = df[df["Name"].isin(st.session_state.players)]
 
                # Apply Action filter if it's not set to "All"
                if Action != "All":
                    filtered_df = filtered_df[filtered_df["Action"].str.contains(Action, case=False, na=False)]
 
                # Apply location filter if it's not "unspecified"
                if activity != "unspecified":
                    filtered_df = filter_by_activity(
                        df=filtered_df,
                        day_night=activity if activity in ["Day", "Night"] else None,
                        environment=activity if activity in ["Outdoor", "Indoor"] else None,
                        distance=activity if activity in ["Close", "Far"] else None
                    )
 
                if no_of_faces > 0:
                    filtered_df = filter_by_no_of_faces(filtered_df, no_of_faces)
 
                if start_date and end_date:
                    filtered_df = filter_by_date(filtered_df, from_date=start_date, to_date=end_date)
 
                # Filter to include only players with the same URL (if applicable)
                filtered_df = filter_by_same_url(filtered_df)
 
                # Ensure only unique URLs in the session state
                st.session_state.filtered_urls = filtered_df["URL"].drop_duplicates().tolist()
 
                if not st.session_state.filtered_urls:
                    st.warning("No images match the selected filters. Showing all images for selected players.")
                    st.session_state.filtered_urls = df[df["Name"].isin(st.session_state.players)]["URL"].drop_duplicates().tolist()
            else:
                st.warning("No players selected or no CSV uploaded.")
                st.session_state.filtered_urls = []
 
        # Add custom CSS for the yellow button
        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                background-color: #FFD700;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                cursor: pointer;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: background-color 0.3s ease, transform 0.2s ease;
            }
            div.stButton > button:first-child:hover {
                background-color: #FFC107;
                transform: scale(1.05);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
       
        def convert_to_drive_direct_view_url(url):
            """Converts a Google Drive file link to a direct download link."""
            try:
                if "drive.google.com/file/d/" in url:
                    # Extract the file ID from the link
                    file_id = url.split("/file/d/")[1].split("/")[0]
                    # Create the direct download link
                    return f"https://drive.google.com/uc?id={file_id}"
                else:
                    st.error(f"Not a valid Google Drive file URL: {url}")
                    return None
            except Exception as e:
                st.error(f"Failed to process Google Drive URL: {url}")
                st.write(f"Error: {e}")
                return None
       
       
        def fetch_image_with_retry(direct_url, retries=3, delay=5):
            for attempt in range(retries):
                try:
                    response = requests.get(direct_url, timeout=5)
                    if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
                        return response.content  # Return the image content
                    else:
                        raise Exception(f"Invalid image response: {response.status_code}")
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(delay)  # Wait before retrying
                        continue
                    else:
                        st.error(f"Failed to load image after {retries} attempts: {e}")
                        return None
       
        # Display images in a grid layout
        if "filtered_urls" in st.session_state and st.session_state.filtered_urls:
            start_index = st.session_state.display_index
            end_index = start_index + 6
            urls_to_display = st.session_state.filtered_urls[start_index:end_index]
            cols = st.columns(3)  # 3 images per row
            for i, url in enumerate(urls_to_display):
                with cols[i % 3]:
                    try:
                        if "drive.google.com" in url:
                            view_link, direct_link = get_drive_view_url_and_direct_link(url)
                            if direct_link:
                                image_content = fetch_image_with_retry(direct_link)
                                if image_content:
                                    image = Image.open(io.BytesIO(image_content))
                                    st.image(image, use_container_width=True)
                                    if view_link:
                                        st.markdown(f'<a href="{view_link}" target="_blank" style="color: blue;">Open in Google Drive</a>', unsafe_allow_html=True)
                        else:
                            st.image(url, use_container_width=True)
                            st.markdown(f'<a href="{url}" target="_blank" style="color: blue;">{url}</a>', unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Failed to load image from URL: {url}")
                        st.write(e)
            # Load More Button
            if end_index < len(st.session_state.filtered_urls):
                if st.button("Load More"):
                    st.session_state.display_index = end_index
       
        # Add some custom CSS to adjust the spacing and alignment
        st.markdown(
            """
        <style>
            .stSelectbox, .stDateInput {
                background-color: #e3f2fd;
                border-radius: 15px;
                border: 1px solid #90caf9;
                padding: 8px 12px;
                font-size: 15px;
            }
        </style>
            """,
            unsafe_allow_html=True,
        )
       
        # Style section
        st.markdown(
            """
        <style>
        .image-gallery {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
        }
        .image-gallery img {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            margin: 10px;
        }
        .center-buttons {
            text-align: center;
            margin-top: 20px;
        }
        .yellow-button {
            background-color: #FFD700;
            border: none;
            padding: 10px 20px;
            color: white;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
        }
        .red-button {
            background-color: #FF4500;
            border: none;
            padding: 10px 20px;
            color: white;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
        }
        .yellow-button:hover {
            background-color: #FFC107;
        }
        .red-button:hover {
            background-color: #FF6347;
        }
        </style>
            """,
            unsafe_allow_html=True,
        )