import streamlit as st
st.set_page_config(page_title='Shorts maker', 
                    page_icon=None, 
                    layout="wide", 
                    initial_sidebar_state="collapsed", 
                    menu_items=None)
from pytube import cipher
class CustomCipher(cipher.Cipher):
    def __init__(self, js: str):
        from typing import List
        import re
        from pytube.exceptions import RegexMatchError
        self.transform_plan: List[str] = cipher.get_transform_plan(js)
        var_regex = re.compile(r"^[\w\$_]+[^\w\$_]")
        var_match = var_regex.search(self.transform_plan[0])
        if not var_match:
            raise RegexMatchError(
                caller="__init__", pattern=var_regex.pattern
            )
        var = var_match.group(0)[:-1]
        self.transform_map = cipher.get_transform_map(js, var)
        self.js_func_patterns = [
            r"\w+\.(\w+)\(\w,(\d+)\)",
            r"\w+\[(\"\w+\")\]\(\w,(\d+)\)"
        ]
        self.throttling_plan = cipher.get_throttling_plan(js)
        self.throttling_array = cipher.get_throttling_function_array(js)
        self.calculated_n = None
cipher.Cipher = CustomCipher
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
import os
from vertexai.generative_models import Part
import json
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from streamlit.runtime.scriptrunner import add_script_run_ctx
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
SAFETY_DEFAULT = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

SAFETY_TEXT = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

BUCKET_ROOT = "jk-content"
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=NeY7kOilSkc"
CUT_TIME = 1
THREAD_SIZE = 8

NUM_OF_TILES_PER_ROW = 5

@st.cache_resource
def get_bucket():
    from google.cloud import storage
    storage_client = storage.Client()
    return storage_client.bucket(BUCKET_ROOT)

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

def split_video(video_id, length):
    contents = [Part.from_uri(uri=f"gs://{BUCKET_ROOT}/{video_id}.mp4", mime_type="video/mp4"),
                f"Split the video more than 20 small videos including short 'title' in 'Korean'\n",
                "Return the result in the JSON array format with keys as follows : \"start_time\", \"title\".\n",
                "The \"start_time\" format is look like this. \"MM:SS\"\n",
                "Example: \n",
                '[{"start_time": "00:00", "title": "주제1"}, {"start_time": "00:10", "title": "주제2"}]']
    response = analyze_gemini(contents, response_mime = 'application/json', isStream=False)
    df = pd.DataFrame(json.loads(response)).rename(columns={"start_time": "start", "title": "title"})
    #debug_container.dataframe(df)
    df['start'] = pd.to_datetime(df['start'], format="%M:%S")
    df['start'] = pd.to_timedelta(df['start'].dt.time.astype(str)).dt.total_seconds()
    df['end'] = df['start'].shift(-1).fillna(length)
    df['duration'] = df['end'] - df['start']
    df = df[['start', 'end', 'duration', 'title']]
    df = df[df['duration'] > (CUT_TIME-1)]
    return df

def parallel_solve(video_id, idx, start, end):
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    outfile = f"{video.video_id}-{start}-{end}.mp4"
    ffmpeg_extract_subclip(f"/cache/{video_id}.mp4", start, end, targetname=f"/cache/{outfile}")
    blob = get_bucket().blob(outfile)
    blob.upload_from_filename(f"/cache/{outfile}")
    return idx, f"/cache/{outfile}"

def load_cache(video_id):
    try:
        df = pd.read_csv(f"/cache/shorts_{video_id}.csv")
    except FileNotFoundError:
        df = pd.DataFrame()
    return df

def save_cache(video_id, df):
    df.to_csv(f"/cache/shorts_{video_id}.csv", index=False, header=True)

def rendering_thumbs(videos_df):
    async_jobs = []
    for idx, row in videos_df.iterrows():
        async_jobs.append((parallel_solve, (video.video_id, idx, row[0], row[1])))

    processed_jobs = []
    with ThreadPoolExecutor(max_workers=THREAD_SIZE) as executor:
        for thread in executor._threads:
            add_script_run_ctx(thread)
        for j in async_jobs:
            processed_jobs.append(executor.submit(j[0], *j[1]))

        for future in as_completed(processed_jobs):
            idx, outfile = future.result()
            videos_df.at[idx, 'file'] = outfile
    return videos_df

col_left, col_right = st.columns(2)
thumbs_container = st.container()
debug_container = st.container()

with col_left:
    video_url = st.text_input("YouTube Video URL", DEFAULT_VIDEO_URL)
    video = YouTube(video_url)
    try:
        video.check_availability()
    except VideoUnavailable:
        st.error("Video is unavailable")
        st.stop()

    size = download_video(video)
    shorts_df = load_cache(video.video_id)

    if len(shorts_df) == 0:
        st.video(video_url)
    else:
        debug_container.dataframe(shorts_df, use_container_width=True)

with col_right:
    cols = st.columns(2)
    if cols[0].button("Start analysis", use_container_width=True):
        with st.status("Analyzing... (1/2)", expanded=True) as status:
            st.write(f"Splitting {size} MB, {video.length} secs video using Gemini 1.5 Pro")            
            with ThreadPoolExecutor(max_workers=1) as executor:
                for thread in executor._threads:
                    add_script_run_ctx(thread)
                task = executor.submit(split_video, video.video_id, video.length)

                with cols[1].empty():
                    start_time = time.time()
                    while task.running():
                        elapsed = f"{(time.time() - start_time):.1f}"
                        st.subheader(elapsed)
                        time.sleep(0.1)
                    st.subheader(f":green[{elapsed}]")
                shorts_df = task.result()
                save_cache(video.video_id, shorts_df)
            status.update(label="Analyzing... (2/2)")
            st.write(f"Creating {len(shorts_df)} videos")
            shorts_df = rendering_thumbs(shorts_df)
            save_cache(video.video_id, shorts_df)
            
            status.update(label="Complete", state="complete")

if shorts_df.empty:
    st.stop()

with thumbs_container:
    cols = st.columns(NUM_OF_TILES_PER_ROW)
    for idx, row in enumerate(shorts_df.itertuples()):
        with cols[idx % NUM_OF_TILES_PER_ROW]:
            container = st.container(height=250)
            container.video(row[5])
            container.caption(f":black[{row[4]}] ({row[3]} secs)")