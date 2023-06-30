# Python In-built packages
from pathlib import Path
import PIL
import requests
from time import time
import io

# External packages
import streamlit as st

# # Local Modules
# import settings
# import helper
from utils import visualize

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Die Preparation Defect Detection")

# Sidebar
st.sidebar.header("Image Upload")

source_img = None
# If image is selected
source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'), 
    accept_multiple_files=False
)

col1, col2 = st.columns(2)

with col1:
    try:
        if source_img is None:
            st.markdown("""
            <style>
            .big-font {
                font-size:100px !important;
                text-align:center
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<p class="big-font">( ͡° ͜ʖ ͡°)</p>', unsafe_allow_html=True)
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image",
                        use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

with col2:
    if source_img is None:
        st.markdown("""
        <style>
        .big-font {
            font-size:100px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="big-font">🔎</p>', unsafe_allow_html=True)
    else:
        res_plotted, detection_df = visualize(source_img)
        st.image(res_plotted, caption='Detected Image',
                    use_column_width=True)
        
        filename = f"{str(int(time()))}.png"
        image_data = PIL.Image.fromarray(res_plotted)

        buf = io.BytesIO()
        image_data.save(buf, format='JPEG')
        byte_im = buf.getvalue()

        btn = st.download_button(
                label="Download image",
                data=byte_im,
                file_name=filename,
                mime="image/png"
                )
        try:
            with st.expander("Detection Results"):
                st.dataframe(detection_df, hide_index=True)
                # for box in boxes:
                #     st.write(box.data)
        except Exception as ex:
            # st.write(ex)
            st.write("No image is uploaded yet!")
