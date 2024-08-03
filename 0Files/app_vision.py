import cv2
import time
import os
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import pyttsx3
import utils
from dotenv import load_dotenv

load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=genai_api_key)

def GenerateFileName(dateTime):
    return './Cache_Recordings/FILE-'+dateTime+'.mp4'

def CaptureVideo(dateTime):
    fileName = GenerateFileName(dateTime)
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

def GeneratePromptForVideo(dateTime):
    return f"""
        You are vision assitant to a blind person. Do below actions:
        Analyse everything from video and give each and every detailed information.
        Provided video date and time details in response in dateMonthyear Hour Minute format. 
        Video DateTime: {dateTime} in "dateMonthyear HourMinuteSecond" format.
        """

def GetModelResponseFromVideo(prompt, filename):
    print("Making LLM inference request...")
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content([prompt, filename],
                                    request_options={"timeout": 600})
    return response.text

def uploadFileToCloud(filename: str):
    print(f"Uploading file...")
    video_file = genai.upload_file(path=filename)
    print(f"Completed upload: {video_file.uri}")
    import time

    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    
    return video_file


def deleteVideoFromCloud(videofile):
    genai.delete_file(videofile.name)
    print(f'Deleted {videofile.display_name}.')

def deleteVideoFromLocal(fileName):
    os.remove(fileName)

#Generate speech from the bigger response
def generateInteractiveSpeech(text):
    prompt_template = f""" You are an assistant of blind person. 
    Blind person provide you the detailed text, from that text provide him the short response in one or two sentences.
    Don't include timing details.
    Text : {text}
    """
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt_template).text

    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()
    return response


# Code for the rag system based on the response recived by the video
def get_text_chunks(text, chunkSize, chunkOverlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=genai_api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=genai_api_key)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=genai_api_key)

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain.invoke( {"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    dateTime = utils.getDateTime()
    try:
        localFineName = CaptureVideo(dateTime)
        fileName = uploadFileToCloud(filename=localFineName)
        response = GetModelResponseFromVideo(GeneratePromptForVideo(dateTime), fileName)
        print(response)
    except:
        print("Something wrong happened, Not able to process video.")
    finally:
        deleteVideoFromLocal(localFineName)
        deleteVideoFromCloud(fileName)

    speechText = generateInteractiveSpeech(response)
    print(speechText)
    
    text_chunks = get_text_chunks(response, 100, 10)
    print(text_chunks)
    get_vector_store(text_chunks)
    response = user_input(input("Provide your query: "))
    print("Answer: " + response)

if __name__ == "__main__":
    main()