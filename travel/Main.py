import streamlit as st
st.set_page_config(page_title='Travel maker Demo', 
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
import folium
from streamlit_folium import folium_static
import requests
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
MAPAPI_KEY = "MAP_API_KEY"
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=F2utz6L76D0"
CUT_TIME = 1   #Drop shorts less than n seconds
THREAD_SIZE = 8
LANGUAGE = 'Korean'
NUM_OF_TILES_PER_ROW = 5

@st.cache_resource
def get_bucket():
    from google.cloud import storage
    storage_client = storage.Client()
    return storage_client.bucket(BUCKET_ROOT)

def create_map():
    # Create the map with Google Maps
    map_obj = folium.Map(tiles=None)
    folium.TileLayer("https://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", 
                     attr="google", 
                     name="Google Maps", 
                     overlay=True, 
                     control=True, 
                     subdomains=["mt0", "mt1", "mt2", "mt3"]).add_to(map_obj)
    return map_obj

def geocode_address(address):
    # URL encode the address
    encoded_address = requests.utils.quote(address)

    # Send a request to the Google Maps Geocoding API
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={MAPAPI_KEY}"
    response = requests.get(geocode_url)
    data = response.json()

    # Check the API response status and extract the coordinates
    if data['status'] == 'OK':
        lat = data['results'][0]['geometry']['location']['lat']
        lon = data['results'][0]['geometry']['location']['lng']
        return round(lat, 6), round(lon, 6)
    else:
        return None, None

def add_markers(map_obj, locations, popup_list=None):
    points = []
    if popup_list is  None:
        # Add markers for each location in the DataFrame
        for lat, lon in locations:
            folium.Marker([lat, lon]).add_to(map_obj)
            points.append((lat, lon))
    else:
        for i in range(len(locations)):
            lat, lon = locations[i]
            popup = popup_list[i]
            folium.Marker([lat, lon], popup=popup).add_to(map_obj)
            points.append((lat, lon))

    #Draw route
    #folium.PolyLine(points, color="blue", weight=2, opacity=1).add_to(map_obj)

    # Fit the map bounds to include all markers
    south_west = [min(lat for lat, _ in locations) - 0.02, min(lon for _, lon in locations) - 0.02]
    north_east = [max(lat for lat, _ in locations) + 0.02, max(lon for _, lon in locations) + 0.02]
    map_bounds = [south_west, north_east]
    map_obj.fit_bounds(map_bounds)

    return map_obj

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

def split_video(video_id, length):
    contents = [Part.from_uri(uri=f"gs://{BUCKET_ROOT}/{video_id}.mp4", mime_type="video/mp4"),
                f"Split the video more than 20 small videos including short 'title' in '{LANGUAGE}' and name of 'city' in 'English'.\n",
                "The 'title' should contain the content that best describes the filming location.\n",
                "Return the result in the JSON array format with keys as follows : \"start_time\", \"title\", \"city\"\n",
                "The \"start_time\" format is like this. \"MM:SS\"",
                "Example: \n",
                '[{"start_time": "00:00", "title": "주제1", "city": "Seoul"}, {"start_time": "00:10", "title": "주제2", "city": "Busan"}]']
    response = analyze_gemini(contents, response_mime = 'application/json', isStream=False)
    df = pd.DataFrame(json.loads(response)).rename(columns={"start_time": "start"})
    df['start'] = pd.to_datetime(df['start'], format="%M:%S")
    df['start'] = pd.to_timedelta(df['start'].dt.time.astype(str)).dt.total_seconds()
    df['end'] = df['start'].shift(-1).fillna(length)
    df['duration'] = df['end'] - df['start']
    df = df[['start', 'end', 'duration', 'title', 'city']]
    df = df[df['duration'] > (CUT_TIME-1)]
    return df

def parallel_solve(video_id, idx, start, end, title, city):
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    outfile = f"{video_id}-{start}-{end}.mp4"
    ffmpeg_extract_subclip(f"/cache/{video_id}.mp4", start, end, targetname=f"/cache/{outfile}")
    blob = get_bucket().blob(outfile)
    blob.upload_from_filename(f"/cache/{outfile}")
    contents = [Part.from_uri(uri=f"gs://{BUCKET_ROOT}/{outfile}", mime_type="video/mp4"),
                f"The title of this video is '{title}' and it was filmed in '{city}'.\n",
                "Find all the 'name's of restaurants, shops, stores, hotels, places, airport or locations in 'English' that appear in the video.\n",
                "Predict the location if unable to find clues.\n",
                "Name of city or country is not location so should not be in result.\n"
                "Return the result in the JSON array format like this. [\"name1\", \"name2\"]. If no, return '[]'\n"]
    locations = []
    try:
        response = analyze_gemini(contents, response_mime = 'application/json', isStream=False)
        locations = json.loads(response)
        locations = [f"{location} in {city}" for location in locations]
    except Exception as e:
        pass
    return idx, f"/cache/{outfile}", locations

def normalize_stores(stores):
    contents = [f"Context: \n{stores}\n",
                "Context is name of places list obtained from video recognition which has some invalid information.\n",
                "Remove invalid, duplicated, common nouns or repeated (ex: Seoul in Seoul) name of places.\n",
                "Return the result in the JSON array format like this. [\"name1 in Seoul\", \"name2 in Busan\"]"]
    final_stores = []
    try:
        response = analyze_gemini(contents, response_mime = 'application/json', isStream=False, safety=SAFETY_TEXT)
        final_stores = json.loads(response)
    except Exception as e:
        pass
    return final_stores

def pin_stores(store_names):
    lat_lng_list = []
    for store in store_names:
        lat, lng = geocode_address(store)
        if lat == None:
            continue
        lat_lng_list.append((store, (lat, lng)))

    return pd.DataFrame(lat_lng_list, columns=["address", "lat_lng"])

def load_cache(video_id):
    try:
        df = pd.read_csv(f"/cache/travel_{video_id}.csv")
    except FileNotFoundError:
        df = pd.DataFrame()
    return df

def save_cache(video_id, df):
    df.to_csv(f"/cache/travel_{video_id}.csv", index=False, header=True)

col_left, col_right = st.columns(2)
travel_program_container = st.container(border=1)
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
    travel_df = load_cache(video.video_id)
    
    st.video(video_url)

stores_df = pd.DataFrame()

with col_right:
    cols = st.columns(4)
    with cols[0]:
        bIndex = st.checkbox("Index", value=True)
    with cols[1]:
        bLocation = st.checkbox("Location", value=True)
    if cols[2].button("Start analysis", use_container_width=True):
        with st.status("Analyzing... (1/4)", expanded=True) as status:
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
                    travel_df = task.result()
                    save_cache(video.video_id, travel_df)

            status.update(label="Analyzing... (2/4)")
            if bLocation:
                st.write(f"Extracting locations from {len(travel_df)} videos using Gemini 1.5 Pro")
                processed_jobs = []
                with ThreadPoolExecutor(max_workers=THREAD_SIZE) as executor:
                    for thread in executor._threads:
                        add_script_run_ctx(thread)
                    for idx, row in travel_df.iterrows():
                        processed_jobs.append(executor.submit(parallel_solve, video.video_id, idx, row[0], row[1], row[3], row[4]))
                    
                    for i, future in enumerate(as_completed(processed_jobs), 1):
                        idx, outfile, locations = future.result()
                        travel_df.at[idx, 'file'] = outfile
                        travel_df.at[idx, 'locations'] = ','.join(locations)
                        status.update(label=f"Video {i}/{len(processed_jobs)} processed")
                save_cache(video.video_id, travel_df)

            status.update(label="Analyzing... (3/4)")
            stores = ""
            for idx, row in travel_df.iterrows():
                if pd.isna(row['locations']) == False:
                    for location in row['locations'].split(","):
                        stores += f"{location}\n"

            st.write(f"Normalizing stores using Gemini")
            final_stores = normalize_stores(stores)

            status.update(label="Analyzing... (4/4)")
            st.write(f"Finding {len(final_stores)} stores location using Google Maps")
            stores_df = pin_stores(final_stores)
            
            status.update(label="Complete", state="complete", expanded=False)

if len(travel_df) > 0:
    debug_container.dataframe(travel_df, use_container_width=True)

with thumbs_container:
    cols = st.columns(NUM_OF_TILES_PER_ROW)
    for idx, row in enumerate(travel_df.itertuples()):
        with cols[idx % NUM_OF_TILES_PER_ROW]:
            container = st.container(height=250)
            container.video(row[6])
            container.caption(f":black[{row[4]} / {row[5]}] ({row[3]} secs)")

if stores_df.empty:
    st.stop()

with col_right:
    m = create_map()
    m = add_markers(m, stores_df['lat_lng'], stores_df['address'])
    folium_static(m, 600, 400)

with travel_program_container:
    final_stores_text = ""
    for res in final_stores:
        final_stores_text += f"{res}\n"
    contents = [Part.from_uri(uri=f"gs://{BUCKET_ROOT}/{video.video_id}.mp4", mime_type="video/mp4"),
                f"You are travel agency. Create travel course with title in '{LANGUAGE}' by refer provided video and found locations below.\n",
                f"Locations: \n{final_stores_text}"]
    with st.spinner('Generating travel suggestion'):
        response = analyze_gemini(contents, isStream=True)
    st.write_stream(gemini_stream_out(response))