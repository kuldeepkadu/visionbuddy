import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import pyttsx3
import os

chunk_size = 100
chunk_overlap = 10

def GenerateFileName(dateTime):
    return './Cache_Recordings/FILE-'+dateTime+'.mp4'

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

def GeneratePromptForVideo(dateTime):
    return f"""
        You are vision assitant to a blind person. Do below actions:
        Analyse everything from provided data and give each and every detailed information.
        """

def GetModelResponseFromVideo(prompt, filename, genai_api_key):
    print("Making LLM inference request...")
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content([prompt, filename],
                                    request_options={"timeout": 600})
    
    get_vector_store(get_text_chunks(response.text, chunk_size, chunk_overlap), genai_api_key)
    return response.text

def deleteVideoFromCloud(videofile):
    genai.delete_file(videofile.name)
    print(f'Deleted {videofile.display_name}.')

def deleteVideoFromLocal(fileName):
    os.remove(fileName)

# Response to user
def generateInteractiveSpeech(text):
    prompt_template = f""" You are an assistant of blind person. 
    Blind person provide you the detailed text, summarize detailed text into short response in one sentence.
    Don't include anything extra.
    Text : {text}
    """
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt_template).text

    file = open(r"response.txt","w+")
    file.write(response)

    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()
    return response


# Code for the rag system based on the response recived by the video
def get_text_chunks(text, chunkSize, chunkOverlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks, genai_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=genai_api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(genai_api_key):
    prompt_template = """
    Answer the question from the provided context, make sure to provide all the details, if the answer is not in
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

def user_input(user_question, genai_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=genai_api_key)

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(genai_api_key)

    response = chain.invoke( {"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]