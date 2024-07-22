import streamlit as st
st.set_page_config(page_title='Keynote summarizer', 
                    page_icon=None, 
                    layout="wide", 
                    initial_sidebar_state="collapsed", 
                    menu_items=None)
from pytubefix import YouTube
import os
import time
from concurrent.futures import ThreadPoolExecutor
from streamlit.runtime.scriptrunner import add_script_run_ctx
from vertexai.generative_models import Part
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
SAFETY_DEFAULT = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

BUCKET_ROOT = "jk-content"
DEFAULT_PROMPT = '영상은 Google I/O 2024에서 촬영된 것입니다. 영상의 내용을 참고하여 제목을 출력하고, 내용을 요약해 주세요. 주요 내용은 별도로 그룹화 하여 설명해 주세요.'
VIDEOS = ['https://www.youtube.com/watch?v=uFroTufv6es',
          'https://www.youtube.com/watch?v=NVwUMyYuLtw',
          'https://www.youtube.com/watch?v=tGENHSG2lWE']

@st.cache_resource
def get_bucket():
    from google.cloud import storage
    storage_client = storage.Client()
    return storage_client.bucket(BUCKET_ROOT)

def gemini_stream_out(responses):
    for response in responses:
        yield response.text

def analyze_gemini(contents, model_name = "gemini-1.5-pro-001", instruction = None, 
                   response_mime = 'text/plain', isStream = False, safety = SAFETY_DEFAULT):
    def get_model():
        if instruction:
            model = GenerativeModel(model_name, system_instruction=instruction)
        else:
            model = GenerativeModel(model_name)
        _ = model._prediction_client
        return model
    
    generation_config={
        "candidate_count": 1,
        "max_output_tokens": 8192,
        "temperature": 0,
        "top_p": 0.5,
        "top_k": 1
    }
    if response_mime != 'text/plain':
        generation_config['response_mime_type'] = response_mime

    #Need to have retry
    for i in range(3):
        try:
            responses = get_model().generate_content(
                contents=contents,
                generation_config=generation_config,
                safety_settings=safety,
                stream=isStream
            )
        except Exception as e:
            import time
            time.sleep(1)
            continue

    if isStream == True:
        return responses

    return responses.text

def download_video(video):
    filename = f"{video.video_id}.mp4"
    if os.path.exists(f"/cache/{filename}") == False:
        video.streams.filter(progressive=True).last().download("/cache", filename=filename)
    blob = get_bucket().blob(filename)
    if blob.exists() == False:
        blob.upload_from_filename(f"/cache/{filename}")
    return int(os.path.getsize(f"/cache/{filename}") / (1024 * 1024))

def gemini_stream_out(responses):
    for response in responses:
        yield response.text

def summarize_video(video_id, prompt, out_container):
    contents = [Part.from_uri(uri=f"gs://{BUCKET_ROOT}/{video_id}.mp4", mime_type="video/mp4"),
                prompt]
    responses = analyze_gemini(contents, isStream=True)
    out_container.write_stream(gemini_stream_out(responses))

tool_container = st.columns([1, 4, 1])
prompt = tool_container[1].text_input("프롬프트", DEFAULT_PROMPT, label_visibility="collapsed")

summarize_container = st.columns(len(VIDEOS))
text_out_container = []
for id, video_url in enumerate(VIDEOS):
    with summarize_container[id].container(border=1):
        st.empty()
        cols = st.columns([1, 2])
        cols[0].video(VIDEOS[id], loop=True, autoplay=True, muted=True)
        text_out_container.append(cols[1])

code_container = st.columns([1, 4, 1])
code_container[1].code('''def summarize_video(video_id, prompt, out_container):
    contents = [Part.from_uri(uri=f"gs://{BUCKET_ROOT}/{video_id}.mp4", mime_type="video/mp4"),
                prompt]
    responses = analyze_gemini(contents, isStream=True)
    out_container.write_stream(gemini_stream_out(responses))''', "python")

async_jobs = []
for id, video_url in enumerate(VIDEOS):
    video = YouTube(video_url)
    size = download_video(video)
    async_jobs.append((summarize_video, (video.video_id, prompt, text_out_container[id],)))

processed_jobs = []
with ThreadPoolExecutor(max_workers=len(VIDEOS)) as executor:    
    for j in async_jobs:
        processed_jobs.append(executor.submit(j[0], *j[1]))
        for thread in executor._threads:
            add_script_run_ctx(thread)

    with tool_container[2].empty():
        start_time = time.time()
        while len(processed_jobs) > 0:
            for job in processed_jobs:
                if job.done():
                    processed_jobs.remove(job)
            elapsed = f"{(time.time() - start_time):.1f}"
            st.subheader(elapsed)
            time.sleep(0.1)
        st.subheader(f":green[{elapsed}]")