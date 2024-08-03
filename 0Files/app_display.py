import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import time
import os

# Directory to save video segments
save_dir = "video_segments"
os.makedirs(save_dir, exist_ok=True)

recording = False
video_counter = 0
video_buffer = []

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.start_time = None

    def recv(self, frame):
        global recording, video_counter

        img = frame.to_ndarray(format="bgr24")
        if self.start_time is None:
            self.start_time = time.time()

        self.frames.append(img)
        
        # Check if 4 seconds have passed
        if time.time() - self.start_time >= 4:
            # Save the frames to a video file
            filename = os.path.join(save_dir, f"segment_{video_counter}.mp4")
            height, width, _ = img.shape
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

            for f in self.frames:
                out.write(f)
            out.release()

            # Reset the buffer and start time
            self.frames = []
            self.start_time = None
            video_counter += 1

        return frame

# UI components
st.title("Video Recorder")

webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
)




# class VideoProcessor:
#     def recv(self, frame):
#         dateTime = utils.getDateTime()
#         fileName = app_vision.GenerateFileName(dateTime)
#         cap = cv2.VideoCapture(0)
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(fileName,fourcc, 20.0, (640,480))
#         start_time = time.time()
#         while( int(time.time() - start_time) < capture_duration ):
#             ret, frame = cap.read()
#             if ret==True:
#                 frame = cv2.flip(frame,1)
#                 out.write(frame)
#             else:
#                 break

#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()
#         return fileName


# def main():
#     st.title("VISION BUDDY :)")
    # dateTime = utils.getDateTime()
    # localFineName = app_vision.CaptureVideo(dateTime)
    # fileName = app_vision.uploadFileToCloud(filename=localFineName)
    # try:
    #     response = app_vision.GetModelResponseFromVideo(app_vision.GeneratePromptForVideo(dateTime), fileName)
    #     print(response)
    # except:
    #     print("Something wrong happened, Not able to process video.")
    # finally:
    #     app_vision.deleteVideoFromLocal(localFineName)
    #     app_vision.deleteVideoFromCloud(fileName)

    # speechText = app_vision.generateInteractiveSpeech(response)
    # print(speechText)
    
    # text_chunks = app_vision.get_text_chunks(response, 100, 10)
    # print(text_chunks)
    # app_vision.get_vector_store(text_chunks)
    # response = app_vision.user_input(input("Provide your query: "))
    # print("Answer: " + response)

# if __name__ == "__main__":
#     main()