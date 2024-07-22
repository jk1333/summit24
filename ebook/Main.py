import streamlit as st
st.set_page_config(page_title='E-Book maker', 
                    page_icon=None, 
                    layout="wide", 
                    initial_sidebar_state="collapsed", 
                    menu_items=None)
from pytubefix import YouTube
from pytubefix.exceptions import VideoUnavailable
import os
from vertexai.generative_models import Part
import json
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from streamlit.runtime.scriptrunner import add_script_run_ctx
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
import cv2
from PIL import Image
from datetime import datetime
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
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=3b88Xv7RVUk"
CUT_TIME = 1
THREAD_SIZE = 8
THUMBNAIL = 'timecode'

if 'page' not in st.session_state:
    st.session_state['page'] = 0

@st.cache_resource
def get_bucket():
    from google.cloud import storage
    storage_client = storage.Client()
    return storage_client.bucket(BUCKET_ROOT)

def getFrame(vidcap, milsec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, milsec)
    _, image_bytes = vidcap.read()
    return cv2.cvtColor(image_bytes, cv2.COLOR_RGB2BGR)

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
    for i in range(5):
        try:
            responses = get_model().generate_content(
                contents=contents,
                generation_config=generation_config,
                safety_settings=safety,
                stream=isStream
            )
        except Exception as e:
            import time
            time.sleep(5)
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
    st.session_state['stream'] = cv2.VideoCapture(f"/cache/{filename}")
    return int(os.path.getsize(f"/cache/{filename}") / (1024 * 1024))

def split_video(video_id, length):
    contents = [Part.from_uri(uri=f"gs://{BUCKET_ROOT}/{video_id}.mp4", mime_type="video/mp4"),
                "We are creating e-book for a 6 year old child using the video.\n",
                "Describe every 'moment' into 'Korean' line by line.\n",
                "Return the result in the JSON array format with keys as follows : \"start_time\", \"moment\".\n",
                "The \"start_time\" format is look like this. \"MM:SS\"\n",
                "Example: \n",
                '[{"time": "00:00", "moment": "요약1"}, {"time": "00:10", "moment": "요약2"}]']
    response = analyze_gemini(contents, response_mime = 'application/json', isStream=False)
    df = pd.DataFrame(json.loads(response)).rename(columns={"start_time": "start"})
    df['start'] = pd.to_datetime(df['start'], format="%M:%S")
    df['start'] = pd.to_timedelta(df['start'].dt.time.astype(str)).dt.total_seconds()
    df['end'] = df['start'].shift(-1).fillna(length)
    df['duration'] = df['end'] - df['start']
    df['middle'] = (df['start'] + df['end'])/2
    df = df[['start', 'end', 'duration', 'moment', 'middle']]
    return df

def parallel_solve(video_id, idx, start, end, moment):
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    outfile = f"{video_id}-{start}-{end}.mp4"
    ffmpeg_extract_subclip(f"/cache/{video_id}.mp4", start, end, targetname=f"/cache/{outfile}")
    blob = get_bucket().blob(outfile)
    blob.upload_from_filename(f"/cache/{outfile}")
    contents = [Part.from_uri(uri=f"gs://{BUCKET_ROOT}/{outfile}", mime_type="video/mp4"),
                f"Find the 'timecode' except '00:00' in the video that best describes '{moment}.'\n",
                #f"Find the 'timecode' in the video that best describes '{moment}'.\n",
                "The \"timecode\" format is look like this. \"MM:SS\"\n",
                "Example: \n",
                '{"timecode": "00:17"}']
    try:
        response = analyze_gemini(contents, response_mime = 'application/json', isStream=False)
        time = json.loads(response)["timecode"]
        time = datetime.strptime(time, '%M:%S')
        offset = (time.second + time.minute*60)
    except Exception as e:
        #Use default if not found proper one
        offset = 0
    return idx, start + offset

def load_cache(video_id):
    try:
        df = pd.read_csv(f"/cache/ebook_{video_id}.csv")
    except FileNotFoundError:
        df = pd.DataFrame()
    return df

def save_cache(video_id, df):
    df.to_csv(f"/cache/ebook_{video_id}.csv", index=False, header=True)

col_left, col_right = st.columns(2)
book_container = st.container(border=2)
debug_container = st.container(border=1)

with col_left:
    video_url = st.text_input("YouTube Video URL", DEFAULT_VIDEO_URL, label_visibility="visible")
    video = YouTube(video_url)
    try:
        video.check_availability()
    except VideoUnavailable:
        st.error("Video is unavailable")
        st.stop()
    
    size = download_video(video)
    moment_df = load_cache(video.video_id)
    
    if len(moment_df) == 0:
        st.video(video_url)

with col_right:
    cols = st.columns(4)
    with cols[0]:
        bIndex = st.checkbox("Index", value=True)
    with cols[1]:
        bThumbnail = st.checkbox("Thumbnail", value=True)
    if cols[2].button("Start analysis", use_container_width=True):
        with st.status("Analyzing... (1/2)", expanded=True) as status:
            if bIndex:
                st.write(f"Splitting {size} MB, {video.length} secs video using Gemini 1.5 Pro")
                with ThreadPoolExecutor(max_workers=1) as executor:
                    for thread in executor._threads:
                        add_script_run_ctx(thread)
                    task = executor.submit(split_video, video.video_id, video.length)
                    with cols[3].empty():
                        start_time = time.time()
                        while task.running():
                            elapsed = f"{(time.time() - start_time):.1f}"
                            st.subheader(elapsed)
                            time.sleep(0.1)
                        st.subheader(f":green[{elapsed}]")
                    moment_df = task.result()
                    save_cache(video.video_id, moment_df)

            status.update(label="Analyzing... (2/2)")
            if bThumbnail:
                st.write(f"Finding best image from {len(moment_df)} moment using Gemini 1.5 Pro")
                processed_jobs = []
                with ThreadPoolExecutor(max_workers=THREAD_SIZE) as executor:
                    for thread in executor._threads:
                        add_script_run_ctx(thread)
                    for idx, row in moment_df.iterrows():
                        processed_jobs.append(executor.submit(parallel_solve, video.video_id, idx, row[0], row[1], row[3]))

                    for i, future in enumerate(as_completed(processed_jobs), 1):
                        idx, timecode = future.result()
                        moment_df.at[idx, 'timecode'] = timecode
                        status.update(label=f"Thumbnail {i}/{len(processed_jobs)} created")
                save_cache(video.video_id, moment_df)
            
            status.update(label="Complete", state="complete")

if moment_df.empty == True:
    st.stop()

with book_container:
    button_left, button_right = st.columns(2)
    if button_left.button("Previous", use_container_width=True):        
        if st.session_state['page'] > 0:
            st.session_state['page'] = st.session_state['page'] - 1
    if button_right.button("Next", use_container_width=True):
        if st.session_state['page'] < len(moment_df) - 1:
            st.session_state['page'] = st.session_state['page'] + 1

    cols = st.columns([2, 1])
    with cols[0]:
        t = moment_df.iloc[st.session_state['page']][THUMBNAIL]
        image = Image.fromarray(getFrame(st.session_state['stream'], t * 1000))
        st.image(image)    
    with cols[1]:
        new_title = f"### {moment_df.iloc[st.session_state['page']]['moment']}"
        st.markdown(new_title, unsafe_allow_html=True)
    st.write(
        """<style>
        [data-testid="stHorizontalBlock"] {
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )    

with debug_container:
    page = st.session_state['page']
    moment = moment_df.iloc[page]
    col_left, col_right = st.columns([1, 2])
    col_left.caption(f"Video id: {page}, Image at {int(moment[THUMBNAIL])} second")
    col_left.video(f"/cache/{video.video_id}-{moment['start']}-{moment['end']}.mp4")
    col_right.dataframe(moment_df[['start', 'moment', THUMBNAIL]], use_container_width=True)