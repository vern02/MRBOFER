# Import necessary libraries
import os
import re
import cv2
import joblib
import spotipy
import threading
import numpy as np
import pandas as pd
import firebase_admin
import seaborn as sns
import streamlit as st
import mediapipe as mp
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Import custom functions from Query.py
# from Query import *

# Additional imports for working with files and Keras
from io import BytesIO
from keras.models import load_model
from firebase_admin import credentials, auth
from streamlit_option_menu import option_menu
from spotipy.oauth2 import SpotifyClientCredentials

# # Fetch
# def view_all_data():
#     data = list(collection.find())  # Fetch all documents in the collection
#     return data
    
# configure the Streamlit app
st.set_page_config(page_title="VibezVision", page_icon="🌍", layout="wide")

# Fetch data using a custom query function and convert it to a Pandas DataFrame
# result = view_all_data()

df = pd.read_csv("Stress.csv")

# df = pd.DataFrame(result, columns=[
#     "Age", "Gender", "University", "Department", "Academic Year",
#     "Current CGPA", "Waiver/Scholarship", "Anxiety Label", "Stress Label", "Depression Label"
# ])

# Initialize Firebase
cred = credentials.Certificate("mrbofer-65af7-4e552797743e.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Function to validate email format using regular expressions
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# Function to validate password length (minimum 8 characters)
def is_valid_password(password):
    return len(password) >= 8

# # Load a pre-trained Keras model and label file for emotion detection
# model = load_model("/Users/vernsin/Documents/UCOMS/FER/model.keras")
# label = np.load("/Users/vernsin/Documents/UCOMS/FER/labels.npy")
    
# Load Keras Model
model_1 = os.path.join(os.path.dirname(__file__), 'model.keras')
model = load_model(model_1)

# Load Emotion Labels
label_1 = os.path.join(os.path.dirname(__file__), 'labels.npy')
label = np.load(label_1)

# Initialize Mediapipe for holistic (face and hand landmarks)
holistic = mp.solutions.holistic
holis = holistic.Holistic()

# Spotify credentials setup
SPOTIPY_CLIENT_ID = 'dd9dc593f3c84895b9a0707d44f3c391'
SPOTIPY_CLIENT_SECRET = 'bffa1591c62445c1a7ef5cb6c5b21298'

# Spotify authentication
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, 
                                                                             client_secret=SPOTIPY_CLIENT_SECRET))

# Initialize session state for last detected emotion
if 'last_detected_emotion' not in st.session_state:
    st.session_state.last_detected_emotion = None

# Open the webcam and display it in the Streamlit app
stop_event = threading.Event()

# Function to open the webcam and process frames
def run_camera():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    # Placeholder for the video frame in the Streamlit app
    stframe = st.empty()

    # Create a placeholder for the detected emotion text
    emotion_text = st.empty() 

    while not stop_event.is_set():
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotion from the frame
        detected_emotion = process_frame(frame)

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the frame in the app
        stframe.image(frame, channels="RGB", use_column_width=True)

        emotion_text.write(f"Detected Emotion: {detected_emotion}")
    
    # Release the webcam when done
    cap.release()

# Function to recommend music based on detected emotion
def recommend_music(emotion, track_position=0):
    # Map emotions to Spotify playlist IDs
    emotion_to_playlist = {
        "happy": "spotify:playlist:37i9dQZF1EIgG2NEOhqsD7",
        "sad": "spotify:playlist:37i9dQZF1EIdChYeHNDfK5",
        "angry": "spotify:playlist:37i9dQZF1EIgNZCaOGb0Mi",
        "neutral": "spotify:playlist:0cmibPCZvH9xC579sIGuzI",
    }

    # Get the playlist ID for the given emotion
    playlist_id = emotion_to_playlist.get(emotion.lower(), None)

    if playlist_id:
        # Fetch tracks from the playlist
        tracks = get_tracks_from_playlist(playlist_id)
        
        # Select a track based on the position in the playlist
        if track_position < len(tracks):
            track = tracks[track_position]
            message = f"**{track['Track']}** → {track['Artist']}"  # Display track and artist
            url = track['URL']
        else:
            # Handle end of playlist
            message = "No more tracks available in the playlist."
            url = None
    else:
        # Handle missing emotion
        message = "No playlist found for this emotion."
        url = None

    # Return both message and URL for display
    return message, url

# Function to retrieve tracks from a Spotify playlist
def get_tracks_from_playlist(playlist_id):
    # Fetch playlist tracks from Spotify
    results = spotify.playlist_tracks(playlist_id)
    if results is None:
        st.error("Error retrieving playlist. Please check the playlist ID.")
        return []
    
    # Extract track details from the playlist
    tracks = results.get('items', [])
    track_list = []
    
    for track in tracks:
        # Ensure the track data is available
        if track['track']:
            track_info = {
                'Track': track['track']['name'],
                'Artist': track['track']['artists'][0]['name'],
                'URL': f'<a href="{track["track"]["external_urls"]["spotify"]}" target="_blank">⏯️</a>'  # Correctly accessing the URL
            }
            track_list.append(track_info)
    
    # Create a DataFrame from the track list
    df_tracks = pd.DataFrame(track_list)

    # Displaying the tracks in a table format
    st.markdown("### Tracks Retrieved:")
    st.markdown(df_tracks.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Return the list of tracks
    return track_list

# Define a function to process frames and predict emotion
def process_frame(frm):
    # Flip the frame horizontally for a mirror-like view
    frm = cv2.flip(frm, 1)
    # Process the frame using Mediapipe
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    # Initialize a list to store landmark features
    lst = []

    # Extract face landmarks and normalize based on the first landmark's coordinates
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x) # Normalize x-coordinates
            lst.append(i.y - res.face_landmarks.landmark[1].y) # Normalize y-coordinates

    # Extract left hand landmarks if available
    if res.left_hand_landmarks:
        for i in res.left_hand_landmarks.landmark:
            lst.append(i.x - res.left_hand_landmarks.landmark[8].x) # Normalize x-coordinates
            lst.append(i.y - res.left_hand_landmarks.landmark[8].y) # Normalize y-coordinates
    else:
        # Pad with zeros if left hand landmarks are missing
        lst.extend([0.0] * 42)

    # Extract right hand landmarks if available
    if res.right_hand_landmarks:
        for i in res.right_hand_landmarks.landmark:
            lst.append(i.x - res.right_hand_landmarks.landmark[8].x) # Normalize x-coordinates
            lst.append(i.y - res.right_hand_landmarks.landmark[8].y) # Normalize y-coordinates
    else:
        # Pad with zeros if right hand landmarks are missing
        lst.extend([0.0] * 42)

    # Convert the list to a NumPy array and reshape for model input
    lst = np.array(lst).reshape(1, -1)

    # Ensure the shape matches model input
    expected_feature_size = 1020 # Expected size of features for the model
    current_feature_size = lst.shape[1]
    if current_feature_size < expected_feature_size:
        lst = np.pad(lst, ((0, 0), (0, expected_feature_size - current_feature_size)), 'constant')

    # Predict the emotion using the pre-trained model
    pred = label[np.argmax(model.predict(lst))]

    # Save the detected emotion to the session state
    st.session_state.last_detected_emotion = pred

    # Return the predicted emotion
    return pred

# Load the stress model
# best_model_path = "/Users/vernsin/Documents/UCOMS/FER/Stress_Model.joblib"

best_model_path_1 = os.path.join(os.path.dirname(__file__), 'Stress_Model.joblib')
best_model_path = joblib.load(best_model_path_1)

# if os.path.exists(best_model_path):
#     best_model = joblib.load(best_model_path)
# else:
#     st.error("Best model file not found!")

# Function for making predictions and recommending music
def predict_stress_level(total_score):
    # Define the logic to classify stress levels and recommend playlists
    if total_score <= 13:
        stress_level = "Low Stress"
        playlist_id = "spotify:playlist:37i9dQZF1DX3rxVfibe1L0" # Calm playlist
    elif 14 <= total_score <= 26:
        stress_level = "Moderate Stress"
        playlist_id = "spotify:playlist:37i9dQZF1DWZd79rJ6a7lp" # Relax playlist
    elif 27 <= total_score <= 40:
        stress_level = "High Perceived Stress"
        playlist_id = "spotify:playlist:37i9dQZF1DWSvKsRPPnv5o" # Stress relief playlist
    else:
        stress_level = "Severe Stress"
        playlist_id = "spotify:playlist:37i9dQZF1DXcCnTAt8CfNe" # Deep focus playlist
    
    # Display the stress level
    st.success(f"The user is likely experiencing **{stress_level}**.")

    # Get and display tracks from the playlist based on stress level
    tracks = get_tracks_from_playlist(playlist_id)

    return stress_level

# Function to classify anxiety levels and recommend music playlists
def predict_anxiety_level(total_score):
    # Define the logic to classify anxiety levels and recommend playlists
    if 0 <= total_score <= 4:
        anxiety_level = "Minimal Anxiety"
        playlist_id = "spotify:playlist:37i9dQZF1DWU0ScTcjJBdj" # Peaceful vibes playlist
    elif  5 <= total_score <= 9:
        anxiety_level = "Mild Anxiety"
        playlist_id = "spotify:playlist:37i9dQZF1DWYcDQ1hSjOpY" # Relaxing playlist
    elif 10 <= total_score <= 14:
        anxiety_level = "Moderate Anxiety"
        playlist_id = "spotify:playlist:37i9dQZF1DXaXDsfv6nvZ5" # Anxiety relief playlist
    elif 15 <= total_score <= 21:
        anxiety_level = "Severe Anxiety"
        playlist_id = "spotify:playlist:37i9dQZF1DWUeWRf9H0wip" # Focus and calm playlist
    
    # Display the anxiety level
    st.success(f"The user is likely experiencing **{anxiety_level}**.")

    # Get and display tracks from the playlist based on stress level
    tracks = get_tracks_from_playlist(playlist_id)

    return anxiety_level

# Function to classify depression levels and recommend music playlists
def predict_depression_level(total_score):
    # Define the logic to classify anxiety levels and recommend playlists
    if total_score == 0:
        depression_level = "No Depression"
        playlist_id = "spotify:playlist:37i9dQZF1DX4sWSpwq3LiO"
    elif total_score == 3:
        depression_level = "Minimal Depression"
        playlist_id = "spotify:playlist:37i9dQZF1DX9XIFQuFvzM4"
    elif 7 <= total_score <= 9:
        depression_level = "Mild Depression"
        playlist_id = "spotify:playlist:37i9dQZF1DWVV27DiNWxkR"
    elif 10 <= total_score <= 14:
        depression_level = "Moderate Depression"
        playlist_id = "spotify:playlist:37i9dQZF1DWZqd5JICZI0u" 
    elif 15 <= total_score <= 19:
        depression_level = "Moderately Severe Depression"
        playlist_id = "spotify:playlist:37i9dQZF1DX0FOF1IUWK1W"
    elif 20 <= total_score <= 27:
        depression_level = "Severe Depression"
        playlist_id = "spotify:playlist:37i9dQZF1DX3YSRoSdA634"
    
    # Display the anxiety level
    st.success(f"The user is likely experiencing **{depression_level}**.")

    # Get and display tracks from the playlist based on stress level
    tracks = get_tracks_from_playlist(playlist_id)

    return depression_level

# Sidebar Navigation: Display sidebar only if the user is logged in
if st.session_state.get('logged_in', False):
    selected = option_menu(
        # Navigation title
        "NAVIGATIONS",
        ['Introduction', 'E-Detection', 'Dashboard', 'Self-Test', 'Logout'],
        menu_icon='browser-safari',
        icons=['opencollective', 'emoji-sunglasses-fill', 'file-bar-graph-fill', 'clipboard-pulse', 'door-open'],
        orientation="horizontal",
        # Default selected menu item
        default_index=0,
        styles={
            # Style for normal menu links
            "nav-link": {"font-weight": "bold"},
            # Style for the selected link
            "nav-link-selected": {"font-weight": "bold"}
        }
    )

# Function to validate if input is an email or username
def is_email_or_username(input_value):
    # Check if it's a valid email
    if is_valid_email(input_value):
        return "email"
    else:
        return "username"

# Plot a pie chart of gender distribution in the data
def plot_gender_distribution(df):
    # Count occurrences of each gender
    gender_counts = df['Gender'].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(gender_counts,
            # Use gender names as labels
            labels=gender_counts.index,
            autopct='%1.1f%%',
            startangle=140)
    plt.title('Gender Distribution')

    st.pyplot(plt)

# Plot a heatmap showing correlations between anxiety, stress, and depression
def plot_heatmap(df):
    # Calculate the correlation between factors after encoding labels as numerical values
    correlation = df[['Anxiety Label', 'Stress Label', 'Depression Label']].apply(lambda x: pd.factorize(x)[0]).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation,
                annot=True, # Annotate cells with correlation values
                cmap='coolwarm',
                fmt=".2f",
                square=True,
                linewidths=.5)
    plt.title('Correlation Between Anxiety, Stress, and Depression')

    st.pyplot(plt)

def app():
    # Initialize session state variables if not already set
    if 'signup' not in st.session_state:
        st.session_state.signup = False

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        
    # Login Sectionn
    if not st.session_state.logged_in and not st.session_state.signup:
        # Centered title section
        col1, col2, col3 = st.columns([1.55, 1, 1])
    
        with col2:
            st.title('Vibez Vision')

        # Centered logo section
        col1, col2, col3 = st.columns([1.6, 1, 1])

        with col2:
            # st.image("/Users/vernsin/Documents/UCOMS/FER/Streamlit/Camera.png", width=200)
            st.image("Camera.png", width=200)

        # Login form
        st.subheader("Login to Your Account")
        st.write("---")
        email = st.text_input('**Email Address**')
        password = st.text_input('**Password**', type='password')

        # Toggle for "Remember Me"
        remember_me = st.toggle("**Remember Me**")

        # Centered buttons for Login and Sign-Up
        col1, col2, col3 = st.columns([2, 5, 2])
        with col2:
            if st.button('**Login**'):
                error_message = ""

                # Validate email and password inputs
                if not is_valid_email(email):
                    error_message = "Please enter a valid email address."
                elif not is_valid_password(password):
                    error_message = "Password must be at least 8 characters long."
                else:
                    try:
                        # Authenticate user using Firebase
                        user = auth.get_user_by_email(email)
                        # firebase.auth().signInWithEmailAndPassword(email, password)
                        st.success('Login Successful')
                        st.session_state.logged_in = True

                        # Handle Remember Me functionality
                        if remember_me:
                            # Store the email in session state for future logins
                            st.session_state['remembered_email'] = email
                        else:
                            # Clear the remembered email if "Remember Me" is not checked
                            st.session_state['remembered_email'] = None

                    except Exception as e:
                        st.warning('Login Failed. Please check your credentials.')

                # Display the error message below the button if it exists
                if error_message:
                    st.markdown(
                        f'''
                        <div style="
                            color: red; 
                            font-weight: bold; 
                            margin-bottom: 10px; 
                            margin-right: 55px;
                            display: inline-block;">
                            {error_message}
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

        with col3:
            if st.button('**Sign Up**'):
                # Navigate to sign-up screen
                st.session_state.signup = True

    # Sign-up section
    elif st.session_state.signup:
        col1, col2, col3 = st.columns([1.55, 1, 1])
    
        with col2:
            st.title('Vibez Vision')

        col1, col2, col3 = st.columns([1.6, 1, 1])

        with col2:
            # st.image("/Users/vernsin/Documents/UCOMS/FER/Streamlit/Camera.png", width=200)
            st.image("Camera.png", width=200)

        # Sign-up form
        st.subheader("Create a New Account")
        st.write("---")

        signup_email = st.text_input('**Email Address**')
        signup_password = st.text_input('**Password**', type='password')
        signup_re_enter_password = st.text_input('**Re-enter Password**', type='password')
        username = st.text_input('**Enter Your Username**')

        # Button to create an account
        col1, col2, col3 = st.columns([1.6, 1, 1])

        with col2:
            # Create my account button in the center
            if st.button('**Create my account**'):
                error_message = ""
                
                # Validate inputs for email, password, and username
                if not is_valid_email(signup_email):
                    error_message = "Please enter a valid email address."
                elif not is_valid_password(signup_password):
                    error_message = "Password must be at least 8 characters long."
                elif signup_password != signup_re_enter_password:
                    error_message = "Passwords do not match. Please try again."
                elif not username:
                    error_message = "Username cannot be empty."
                else:
                    try:
                        # Create the user in Firebase
                        user = auth.create_user(
                            email=signup_email,
                            password=signup_password,
                            display_name=username
                        )
                        st.success(f'Account created for {username}!')
                        st.session_state.signup = False
                    except Exception as e:
                        st.warning(f'Error: {e}')

                # Display the error message below the button if it exists
                if error_message:
                    st.markdown(
                        f'''
                        <div style="
                            text-align: center; 
                            color: red; 
                            font-weight: bold; 
                            margin-bottom: 10px; 
                            margin-right: 55px;
                            display: inline-block;">
                            {error_message}
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

        # Use columns to center the buttons
        col1, col2, col3 = st.columns([1.67, 1, 1])
        
        with col2:
            if st.button('**Back to Login**'):
                st.session_state.signup = False

    # Main app functionality after login
    elif st.session_state.logged_in:

        # Main content
        if selected == "Introduction":
            # Display the app introduction section
            st.markdown(
                "<div style='text-align: center; font-size: 32px;'>Welcome to <strong>Vibez Vision</strong></div>", 
                unsafe_allow_html=True
                )
            
            st.write('---')

            st.markdown(
                """
                <div style="
                    text-align: center; 
                    margin: auto; 
                    padding: 10px;">
                    <strong>Vibez Vision</strong> is your personalized emotional wellness platform designed to help you navigate your emotions and enhance your well-being.
                    Using state-of-the-art facial emotion recognition and self-assessment tools, Vibez Vision empowers you with self-awareness and offers tailored 
                    music therapy to uplift your mood and support your mental health journey.
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.write('---')

            # Brief description of app features
            st.write("### **How it works:**")

            st.write("##### ** E-Detection **")
            st.markdown("""
                    E-Detection leverages cutting-edge facial recognition technology to analyze your emotions in real-time. 
                    By enabling your camera, Vibez Vision can detect your dominant emotion—be it joy, sadness, calmness, or excitement. 
                    Based on the identified emotion, the system generates a personalized playlist, providing music therapy to complement or enhance your emotional state. 
                    This seamless integration of technology and wellness empowers you to better understand and manage your feelings.
                        
                        Go To < 😎 E-Detection >
                           """)
            st.write("---")
            
            st.write("##### ** Dashboard **")
            st.markdown("""
                    The Dashboard offers powerful visualizations that provide insights into emotional trends derived from a vast dataset. 
                    Through engaging graphs and analytics, users can explore patterns related to stress, anxiety, and emotional fluctuations.
                    While this section isn't based on real-time user data, it highlights global emotional patterns to promote awareness and understanding.
                        
                        Go To < 📈 Dashboard >
                           """)
            st.write("---")
            
            st.write("##### ** Self-Test **")
            st.markdown("""
                    The Self-Test section empowers you to assess your emotional well-being by evaluating stress, anxiety, and depression levels. 
                    Through interactive questionnaires tailored to each emotional state, Vibez Vision delivers insights into your mental health. 
                    Additionally, it recommends soothing music playlists tailored to your results, offering an effective way to alleviate potential symptoms. 
                    Take charge of your wellness journey by understanding your emotions and accessing personalized music therapy interventions.
                                                
                        Go To < ✅ Self-Test >
                           """)
                
            st.write('---')

            st.markdown(
                "<div style='text-align: center;'><strong>Vibez Vision:</strong> Your companion for emotional wellness, blending self-awareness and music therapy to enhance your well-being.</div>", 
                unsafe_allow_html=True
                )
            
        # Introduction of this music recommendation based on facial emotion recognition
        elif selected == "E-Detection":
            # Emotion detection section
            st.markdown("# 😎 Face Recognition")
            st.write('---')
            st.markdown("""
                    ### How It Works:
                    This section utilizes cutting-edge facial recognition technology to analyze your emotions in real-time. 
                    Simply look into the camera and Vibez Vision will detect your dominant emotion.
                        
                    To use the emotion detection and music recommendation feature, follow these steps:

                        

                        1. Start Emotion Detection:
                            The user begins by clicking the Start button. This activates the camera, allowing it to capture the user's facial expressions in real time.

                        2. Emotion Analysis:
                            While the camera is running, the system analyzes the user's face to detect their current emotional state. The detected emotion is displayed at the bottom of the camera frame, providing real-time
                            feedback on the user's emotional status.

                        3. Stop and Get Recommendations:
                            Once the user has viewed their emotion status, they can click the Stop button to finalize the process. After stopping, the system will generate personalized music recommendations based on the
                            detected emotion. The user will receive a message with a link to the suggested music, tailored to uplift or match their emotional state.
                        
                    **Based on the detected emotion**, personalized music recommendations will be generated to uplift and support your emotional state.

                    #### Key Instructions:
                    - Ensure you are well-lit and facing the camera.
                    - Hold still while the system processes your face.
                    - After detection, music will be recommended based on your emotional state.
                        """)
            st.write('---')

            # Buttons to start/stop the camera
            col1, col2, col3, col4, col5 = st.columns([2,2,2,2,0.6])

            # Place the start button in the first column
            with col1:
                start_detection = st.button('▶️ Start', key='start_button')

            # Place the stop button in the second column
            with col5:
                stop_detection = st.button('⏹️ Stop', key='stop_button')

            # Trigger camera functionality
            if start_detection:
                run_camera()
            
            # Display detected emotion and recommended music
            detected_emotion = st.session_state.last_detected_emotion
            if detected_emotion:
                # Show the detected emotion
                st.write(f"Detected Emotion: **{detected_emotion}**")

                # Call the music recommendation function and display the result
                detected_emotion = st.session_state.last_detected_emotion
                message, url = recommend_music(detected_emotion)
                st.write(message)
                if url:
                    # Provide a link to the recommended music
                    st.markdown(f"[Listen here]({url})")
                else:
                    # Handle cases where no emotion is detected
                    st.write("No emotion detected to play music.")

        # Dashboard for data visualization
        elif selected == 'Dashboard':
            st.title("📈 Dashboard")
            st.write('---')

            # Sidebar with interactive filters
            st.sidebar.markdown(
                """
                <h3 style="display: flex; align-items: center;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-funnel-fill" viewBox="0 0 16 16">
                        <path d="M1.5 1.5A.5.5 0 0 1 2 1h12a.5.5 0 0 1 .374.832L10 6.54V12.5a.5.5 0 0 1-.8.4l-2-1.5a.5.5 0 0 1-.2-.4V6.54L1.126 1.832A.5.5 0 0 1 1.5 1.5z"/>
                    </svg>
                    <span style="margin-left: 5px;">Filters</span>
                </h3>
                """,
                unsafe_allow_html=True
            )

            # Category selection for visualizations
            category = st.sidebar.radio("Select a Category", ["General Distributions", "Levels of Psychological", "Mood/Gender"])

            # Filters for Gender and Department with all departments selected by default
            selected_genders = st.sidebar.multiselect("Filter by Gender",
                                                      options=df['Gender'].unique(),
                                                      default=df['Gender'].unique())
            
            selected_departments = st.sidebar.multiselect("Filter by Department",
                                                          options=df['Department'].unique())
                                         
            # Apply filters to DataFrame, only filter if a selection is made
            filtered_df = df.copy()
            if selected_genders:
                filtered_df = filtered_df[filtered_df['Gender'].isin(selected_genders)]
            if selected_departments:
                filtered_df = filtered_df[filtered_df['Department'].isin(selected_departments)]

            # Customizable color palette
            palette_options = {
                "Default": px.colors.qualitative.Plotly,
                "BrightColors": px.colors.qualitative.Bold,
                "SoftPastels": px.colors.qualitative.Pastel,
                "VintageTones": px.colors.qualitative.Antique,
                "DarkTheme": px.colors.qualitative.Dark24,
                "LightTheme": px.colors.qualitative.Light24,
                "AlphabeticalColors": px.colors.qualitative.Alphabet,
                "ColorBlindFriendly": px.colors.qualitative.Safe,
                "ColorCode2": px.colors.qualitative.Set2,
                "ColorCode3": px.colors.qualitative.D3,

            }
            selected_palette = st.sidebar.selectbox("Select Color Palette", options=list(palette_options.keys()))
            color_palette = palette_options[selected_palette]

            # Mood/Gender Visualizations
            if category == "Mood/Gender":
                st.subheader("Mood and Gender Distribution")
                
                # Button to create an account
                col1, col2, col3 = st.columns([12,0.1,12])

                with col1:
                    # Visualization 1: Box Plot for Anxiety
                    st.markdown(
                        """
                        <div style="
                            border: 2px solid #ccc; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin-bottom: 20px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                            <h4 style="text-align: center;">Anxiety by Gender</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Box Plot for Anxiety
                    fig_box = px.box(filtered_df,
                                    x="Gender",
                                    y="Anxiety Label",
                                    color="Gender",
                                    color_discrete_sequence=color_palette)
                    
                    st.plotly_chart(fig_box, use_container_width=True)
                
                with col3:
                    # Visualization 2: Violin Plot for Stress
                    st.markdown(
                        """
                        <div style="
                            border: 2px solid #ccc; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin-bottom: 20px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                            <h4 style="text-align: center;">Stress by Gender</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Violin Plot for Stress
                    fig_violin = px.violin(filtered_df,
                                        x="Gender",
                                        y="Stress Label",
                                        color="Gender",
                                        box=True,
                                        points="all",
                                        color_discrete_sequence=color_palette)
                    
                    st.plotly_chart(fig_violin, use_container_width=True)
                
                # Visualization 3: Swarm Plot for Depression
                st.markdown(
                    """
                    <div style="
                        border: 2px solid #ccc; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 20px;
                        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                        <h4 style="text-align: center;">Depression by Gender</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Swarm Plot for Depression
                fig_swarm = px.strip(filtered_df,
                                     x="Gender",
                                     y="Depression Label",
                                     color="Gender",
                                     color_discrete_sequence=color_palette)
                
                st.plotly_chart(fig_swarm, use_container_width=True)

            # Display visualizations based on selected category
            elif category == "General Distributions":
                st.subheader("General Distributions")

                # Button to create an account
                col1, col2, col3 = st.columns([8,0.1,12])

                with col1:
                    st.markdown(
                        """
                        <div style="
                            border: 2px solid #ccc; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin-bottom: 20px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                            <h4 style="text-align: center;">Academic Year and Department</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Sunburst Chart for Academic Year and Department
                    fig_sunburst = px.sunburst(filtered_df,
                                            path=['Academic Year', 'Department'],
                                            color='Gender',
                                            color_discrete_sequence=color_palette)
                    
                    st.plotly_chart(fig_sunburst, use_container_width=True)

                with col3:
                    st.markdown(
                        """
                        <div style="
                            border: 2px solid #ccc; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin-bottom: 20px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                            <h4 style="text-align: center;">Gender Distribution Across Departments</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Stacked Bar Chart for Gender and Department
                    fig_bar = px.bar(filtered_df,
                                    x='Department',
                                    color='Gender',
                                    barmode='stack',
                                    color_discrete_sequence=color_palette)
                    
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Label mappings
                depression_label_mapping = {
                    'No Depression': 0,
                    'Minimal Depression': 1,
                    'Mild Depression': 2,
                    'Moderate Depression': 3,
                    'Moderately Severe Depression': 4,
                    'Severe Depression': 5,
                }

                stress_label_mapping = {
                    'Low Stress': 0,
                    'Moderate Stress': 1,
                    'High Perceived Stress': 2,
                    'Severe Stress': 3
                }

                anxiety_label_mapping = {
                    'Minimal Anxiety': 0,
                    'Mild Anxiety': 1,
                    'Moderate Anxiety': 2,
                    'Severe Anxiety': 3
                }

                # Apply the mapping to the labels
                filtered_df['Depression Label'] = filtered_df['Depression Label'].map(depression_label_mapping)
                filtered_df['Anxiety Label'] = filtered_df['Anxiety Label'].map(anxiety_label_mapping)
                filtered_df['Stress Label'] = filtered_df['Stress Label'].map(stress_label_mapping)

                # Handle NaN values
                filtered_df['Depression Label'] = filtered_df['Depression Label'].fillna(0)
                filtered_df['Anxiety Label'] = filtered_df['Anxiety Label'].fillna(0)
                filtered_df['Stress Label'] = filtered_df['Stress Label'].fillna(0)

                # Ensure columns are numeric
                filtered_df['Depression Label'] = pd.to_numeric(filtered_df['Depression Label'], errors='coerce')
                filtered_df['Anxiety Label'] = pd.to_numeric(filtered_df['Anxiety Label'], errors='coerce')
                filtered_df['Stress Label'] = pd.to_numeric(filtered_df['Stress Label'], errors='coerce')

                # Confirm no NaN values remain
                if filtered_df[['Depression Label', 'Anxiety Label', 'Stress Label']].isna().sum().sum() > 0:
                    st.warning("There are still NaN values in the dataset after cleaning.")

                st.markdown(
                        """
                        <div style="
                            border: 2px solid #ccc; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin-bottom: 20px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                            <h4 style="text-align: center;">Psychological Levels</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Bubble Chart for Depression vs Anxiety with Stress as Size
                fig_bubble = px.scatter(
                    filtered_df,
                    x='Depression Label',
                    y='Anxiety Label',
                    size='Stress Label',  # Stress as the size of the bubbles
                    color='Department',   # Assuming 'Department' exists in your DataFrame
                    color_discrete_sequence=color_palette,
                    hover_data={
                        'Depression Label': True, 
                        'Anxiety Label': True, 
                        'Stress Label': True
                    }
                )

                # Display the bubble chart
                st.plotly_chart(fig_bubble, use_container_width=True)

            # Levels of Psychological
            elif category == "Levels of Psychological":
                st.subheader("Levels of Anxiety, Stress, and Depression")

                # Button to create an account
                col1, col2, col3 = st.columns([12,0.1,12])

                with col1:
                    st.markdown(
                        """
                        <div style="
                            border: 2px solid #ccc; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin-bottom: 20px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                            <h4 style="text-align: center;">Anxiety Levels</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Anxiety Levels Histogram
                    fig_anxiety = px.histogram(filtered_df,
                                            x='Anxiety Label',
                                            color='Anxiety Label',
                                            color_discrete_sequence=color_palette)
                    
                    fig_anxiety.update_traces(marker=dict(line=dict(width=1, color="black")))
                    fig_anxiety.update_layout(margin=dict(l=0, r=50, t=50, b=0))

                    st.plotly_chart(fig_anxiety, use_container_width=True)

                with col3:
                    st.markdown(
                        """
                        <div style="
                            border: 2px solid #ccc; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin-bottom: 20px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                            <h4 style="text-align: center;">Stress Levels</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Stress Levels Histogram
                    fig_stress = px.histogram(filtered_df,
                                            x='Stress Label',
                                            color='Stress Label',
                                            color_discrete_sequence=color_palette)
                    
                    fig_stress.update_traces(marker=dict(line=dict(width=1, color="black")))
                    fig_stress.update_layout(margin=dict(l=0, r=50, t=50, b=0))

                    st.plotly_chart(fig_stress, use_container_width=True)

                st.markdown(
                        """
                        <div style="
                            border: 2px solid #ccc; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin-bottom: 20px;
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                            <h4 style="text-align: center;">Depression Levels</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Depression Levels Histogram
                fig_depression = px.histogram(filtered_df,
                                              x='Depression Label',
                                              color='Depression Label',
                                              color_discrete_sequence=color_palette)
                
                fig_depression.update_traces(marker=dict(line=dict(width=1, color="black")))
                fig_depression.update_layout(margin=dict(l=0, r=50, t=50, b=0))

                st.plotly_chart(fig_depression,
                                use_container_width=True)

            # Download filtered data option
            st.sidebar.download_button(
                label="Download Filtered Data",
                data=filtered_df.to_csv(index=False),
                file_name="filtered_data.csv",
                mime="text/csv"
            )

        # Self-Test section
        elif selected == "Self-Test":
            st.markdown("# ✅ Self-Test")
            st.write('---')
            # Display an instructional header for self-test
            st.markdown(
                "<div style='text-align: center; font-size: 22px;'><strong>* Please click on the button in the sidebar to access the self-test options and chose the one that best suits you. *</strong></div>", 
                unsafe_allow_html=True
                )
            # st.markdown("#### * Please click on the button in the sidebar to access the self-test options and chose the one that best suits you. *")
            
            st.write('---')

            # Provide an overview of how the self-test works
            st.markdown("""
                        ##### How It Works:
                        This section offers various self-assessment tools to gauge your stress, anxiety, and depression levels.
                        Answer a series of questions tailored to each emotional state and receive personalized insights into your well-being.
                        Additionally, **Vibez Vision** recommends soothing music playlists curated to alleviate potential symptoms associated with stress, anxiety, or depression.
                        """)
            
            st.write('---')
            
            # Sidebar options to select the type of self-test
            st.sidebar.write("## Prediction Type")
            if "selected_test" not in st.session_state:
                st.session_state.selected_test = None

            # Buttons to choose between stress, anxiety, or depression prediction
            if st.sidebar.button("Stress Prediction"):
                st.session_state.selected_test = "Stress"

            if st.sidebar.button("Anxiety Prediction"):
                st.session_state.selected_test = "Anxiety"

            if st.sidebar.button("Depression Prediction"):
                st.session_state.selected_test = "Depression"

            # Stress Prediction Form
            if st.session_state.selected_test == "Stress":
                st.markdown("# Stress Prediction")
                st.write("##### Please rate the following questions from 0 to 4:")

                st.write("")
                st.write("")

                # Collect user inputs via sliders
                q1 = st.slider("1. In a semester, how often have you felt upset due to something that happened in your academic affairs?", 0, 4, 0)
                q2 = st.slider("2. In a semester, how often you felt as if you were unable to control important things in your academic affairs?", 0, 4, 0)
                q3 = st.slider("3. In a semester, how often you felt nervous and stressed because of academic pressure?", 0, 4, 0)
                q4 = st.slider("4. In a semester, how often you felt as if you could not cope with all the mandatory academic activities? (e.g, assignments, quiz, exams)", 0, 4, 0)
                q5 = st.slider("5. In a semester, how often you felt confident about your ability to handle your academic / university problems?", 0, 4, 0)
                q6 = st.slider("6. In a semester, how often you felt as if things in your academic life is going on your way?", 0, 4, 0)
                q7 = st.slider("7. In a semester, how often are you able to control irritations in your academic / university affairs?", 0, 4, 0)
                q8 = st.slider("8. In a semester, how often you felt as if your academic performance was on top?", 0, 4, 0)
                q9 = st.slider("9. In a semester, how often you got angered due to bad performance or low grades that is beyond your control?", 0, 4, 0)
                q10 = st.slider("10. In a semester, how often you felt as if academic difficulties are piling up so high that you could not overcome them?", 0, 4, 0)

                # Calculate total score for stress prediction
                total_score = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10

                if st.button("Predict Stress Level"):
                    if best_model:  # Ensure the model is loaded
                        # Predict stress level
                        stress_level = predict_stress_level(total_score)
                    else:
                        st.error("Stress model not loaded properly.")

            # Depression Prediction Form
            elif st.session_state.selected_test == "Depression":
                st.markdown("# Depression Prediction")
                st.write("##### Please rate the following questions from 0 to 4:")

                st.write("")
                st.write("")

                # Collect user inputs via sliders
                q1 = st.slider("1. In a semester, how often have you had little interest or pleasure in doing things?", 0, 4, 0)
                q2 = st.slider("2. In a semester, how often have you been feeling down, depressed or hopeless?", 0, 4, 0)
                q3 = st.slider("3. In a semester, how often have you had trouble falling or staying asleep, or sleeping too much?", 0, 4, 0)
                q4 = st.slider("4. In a semester, how often have you been feeling tired or having little energy?", 0, 4, 0)
                q5 = st.slider("5. In a semester, how often have you had poor appetite or overeating?", 0, 4, 0)
                q6 = st.slider("6. In a semester, how often have you been feeling bad about yourself - or that you are a failure or have let yourself or your family down?", 0, 4, 0)
                q7 = st.slider("7. In a semester, how often have you been having trouble concentrating on things, such as reading the books or watching television?", 0, 4, 0)
                q8 = st.slider("8. In a semester, how often have you moved or spoke too slowly for other people to notice? Or you've been moving a lot more than usual because you've been restless?", 0, 4, 0)
                q9 = st.slider("9. In a semester, how often have you had thoughts that you would be better off dead, or of hurting yourself?", 0, 4, 0)

                # Calculate total score for stress prediction
                total_score = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9

                if st.button("Predict Depression Level"):
                    if best_model:  # Ensure the model is loaded
                        # Predict stress level
                        depression_level = predict_depression_level(total_score)
                    else:
                        st.error("Depression model not loaded properly.")

            # Anxiety Prediction Form
            elif st.session_state.selected_test == "Anxiety":
                st.markdown("# Anxiety Prediction")
                st.write("##### Please rate the following questions from 0 to 4:")

                st.write("")
                st.write("")

                # Collect user inputs via sliders
                q1 = st.slider("1. In a semester, how often you felt nervous, anxious or on edge due to academic pressure?", 0, 4, 0)
                q2 = st.slider("2. In a semester, how often have you been unable to stop worrying about your academic affairs?", 0, 4, 0)
                q3 = st.slider("3. In a semester, how often have you had trouble relaxing due to academic pressure?", 0, 4, 0)
                q4 = st.slider("4. In a semester, how often have you been easily annoyed or irritated because of academic pressure?", 0, 4, 0)
                q5 = st.slider("5. In a semester, how often have you worried too much about academic affairs?", 0, 4, 0)
                q6 = st.slider("6. In a semester, how often have you been so restless due to academic pressure that it is hard to sit still?", 0, 4, 0)
                q7 = st.slider("7. In a semester, how often have you felt afraid, as if something awful might happen?", 0, 4, 0)
                
                # Calculate total score for stress prediction
                total_score = q1 + q2 + q3 + q4 + q5 + q6 + q7 

                if st.button("Predict Anxiety Level"):
                    if best_model:  # Ensure the model is loaded
                        # Predict stress level
                        anxiety_level = predict_anxiety_level(total_score)
                    else:
                        st.error("Anxiety model not loaded properly.")
                        anxiety_level = None

        elif selected == "Logout":
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.session_state.user_email = None

if __name__ == "__main__":
    app()
