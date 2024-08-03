import cv2
import time
import vertexai
import google.generativeai as genai
import google.ai.generativelanguage as glm
from vertexai.generative_models import GenerativeModel, Image, Part

GOOGLE_API_KEY = "AIzaSyDfr1zvUZF9bLdyuOJjV_pg48-L_vXtW6g"
PROJECT_ID = "276310965140"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}


def GenerateFileName():
    return 'FILE-'+time.strftime("%d%b%Y %H%M%S",time.gmtime())+'.mp4'

def CaptureVideo():
    fileName = GenerateFileName()
    # The duration in seconds of the video captured
    capture_duration = 2

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(fileName,fourcc, 20.0, (640,480))
    start_time = time.time()
    while( int(time.time() - start_time) < capture_duration ):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,1)
            out.write(frame)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return fileName

# For capturing the images through camera access
def CaptureImage():
    videoCaptureObject = cv2.VideoCapture(0)
    result = True
    while(result):
        ret,frame = videoCaptureObject.read()
        cv2.imwrite("NewPicture.jpg",frame)
        result = False
    videoCaptureObject.release()
    cv2.destroyAllWindows()

def GeneratePromptForVideo():
    return """You are vision assitant to a blind person.
        Understand the video and help the person to know what is present there in one to two sentences
        """

def EnableModelConnection():
    genai.configure(api_key=GOOGLE_API_KEY)
    # Initialize Vertex AI
    # vertexai.init(project=PROJECT_ID, location=LOCATION)
    return genai.GenerativeModel("gemini-pro-vision")

def GetModelResponseFromVideo(prompt, model, filename):
    # video = Part.from_uri(filename, mime_type="video/mp4")
    contents = [prompt, filename]
    responses = model.generate_content(contents, stream=True)
    response = ""
    for response in responses:
        response += response.text
    return response


def main():
    fileName = CaptureVideo()
    print(fileName)
    model = EnableModelConnection()
    response = GetModelResponseFromVideo(GeneratePromptForVideo(), model, fileName)
    print(response)

if __name__ == "__main__":
    main()