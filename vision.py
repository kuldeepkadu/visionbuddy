import google.generativeai as genai
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import time
import os
import threading
import queue
from dotenv import load_dotenv
import gemini_api
import utils

load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=genai_api_key)

video_counter = 0
CAPTURE_DURATION = 4
video_buffer = []
segment_queue = queue.Queue()

# Initialize chat session in Streamlit if not already present
if "messages" not in st.session_state:
    st.session_state['messages'] = []

# Initialize text display state in Streamlit if not already present
if "displayed_text" not in st.session_state:
    st.session_state['displayed_text'] = ""

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.start_time = None

    def recv(self, frame):
        global video_counter

        img = frame.to_ndarray(format="bgr24")
        if self.start_time is None:
            self.start_time = time.time()

        self.frames.append(img)
        dateTime = utils.getDateTime()
        # Check if 4 seconds have passed
        if time.time() - self.start_time >= CAPTURE_DURATION:
            # Save the frames to a video file
            filename = os.path.join(gemini_api.GenerateFileName(dateTime=dateTime))
            height, width, _ = img.shape
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

            for f in self.frames:
                out.write(f)
            out.release()

            # Put the filename into the queue for processing
            segment_queue.put(filename)

            # Reset the buffer and start time
            self.frames = []
            self.start_time = None
            video_counter += 1

        return frame

def process_segments():
    while True:
        dateTime = utils.getDateTime()
        # Get the next segment from the queue
        localFilename = segment_queue.get()
        print("Started Processing FileName: ", localFilename)
        if localFilename is None:
            break
        
        # Process the segment (e.g., apply some analysis)
        try:
            cloudFilePath = gemini_api.uploadFileToCloud(filename=localFilename)
            response = gemini_api.GetModelResponseFromVideo(gemini_api.GeneratePromptForVideo(dateTime), cloudFilePath, genai_api_key)
            print(response)
        except Exception as e:
            print("Something wrong happened, Not able to process video.", e)
            response = "Error occurred during processing."
        finally:
            gemini_api.deleteVideoFromLocal(localFilename)
            gemini_api.deleteVideoFromCloud(cloudFilePath)

        gemini_api.generateInteractiveSpeech(response)

def typewriter(container, text: str, speed: int):
    tokens = text.split()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(f"<h5 style='color: black;'>{curr_full_text}</h5>", unsafe_allow_html=True)
        time.sleep(1 / speed)

def main():
    # UI components
    st.set_page_config(layout="wide", page_icon="eye", page_title="Vision Buddy")
    st.markdown("<h1 style='text-align: center; color: black;'><u>Vision Buddy</u></h1>", unsafe_allow_html=True)

    # Start the processing thread
    processing_thread = threading.Thread(target=process_segments, daemon=True)
    processing_thread.start()

    col1, col2, col3 = st.columns([10.9, 0.2, 10.9])

    with col1:
        webrtc_ctx = webrtc_streamer(
            key="Vision Buddy",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=VideoProcessor,
        )

        container = st.empty()

        def typewriter_thread():
            speed = 15
            while True:
                typewriter(container, text=open('response.txt', 'r').read().rstrip(), speed=speed)

        # Start the typewriter effect in a separate thread
        typewriter_thread = threading.Thread(target=typewriter_thread, daemon=True)
        typewriter_thread.start()

    with col2:
        st.html(
            '''
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border-left: 2px solid rgba(49, 51, 63, 0.2);
                        height: 60vh;
                        margin: auto;
                    }
                </style>
            '''
        )

    with col3:
        container = st.container(height=400, border=0)
        user_question = st.chat_input("Ask the question")
        if user_question:
            st.session_state['messages'].append(user_question)
            st.session_state['messages'].append(gemini_api.user_input(user_question, genai_api_key))

            with container:
                for i in range(len(st.session_state['messages'])):
                    if (i % 2) == 0:
                        with st.chat_message("user"):
                            st.write(st.session_state['messages'][i])
                    else:
                        with st.chat_message("assistant"):
                            st.write(st.session_state['messages'][i])

if __name__ == "__main__":
    main()
