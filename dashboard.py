import streamlit as st

from streamlit_drawable_canvas import st_canvas
from PIL import Image

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import segmentation_models_pytorch as smp


def load_model(fp):
    print("Start loading the model...")
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    best_model = torch.load(fp, map_location=torch.device('cpu'))
    model.load_state_dict(best_model)
    print("Model loaded!")
    return model


def preprocess_img(image):
    print("Start preprocessing image...")
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # resize to a fixed size
        transforms.ToTensor(),  # convert to tensor (normalize between 0 and 1)
    ])
    image = transform(image).unsqueeze(0)
    print("Image preprocessed!")
    return image


def predict_mask(model, image):
    print("Start prediction...")
    model.eval()
    with torch.no_grad():
        output = model(image)
    print("Prediction made!")
    pred = torch.sigmoid(output) > 0.5
    return pred[0].squeeze().numpy()


def post_process_mask(pred_mask):
    pred_mask = pred_mask.astype(np.uint8) * 255  # convert to binary image with 0 and 255
    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)  # CLOSING = DILATION + EROSION --> to close small holes in foreground objects
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)   # OPENING = EROSION + DILATION --> to remove noise

    # Find contours
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw polygons
    post_processed_mask = np.zeros_like(pred_mask)

    # Approximate contours with polygons and draw them on the mask
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(post_processed_mask, [approx], -1, 255, thickness=cv2.FILLED)

    return (post_processed_mask // 255).astype(bool)


# function to enable file uploader when the uploaded file is deleted
def enable():
    st.session_state.disabled = False


st.set_page_config(layout="wide")

# Push the file uploader and the selectbox more down and set the background image
st.markdown(
    """<style>
            .st-emotion-cache-up131x, .st-emotion-cache-ue6h4q {
                margin-top: 50px;
            }
            .st-emotion-cache-1jicfl2 {
                padding-top: 20px;
            }
            
            /* Set the background image */
            [data-testid="stAppViewContainer"] {
                background-image: url("https://www.shutterstock.com/image-vector/white-map-city-curtiba-digital-600nw-1339788473.jpg");
                background-size: cover;  /* This makes the image cover the entire screen */
                background-position: center;
                background-repeat: no-repeat;
                height: 100vh;  /* Ensures the element takes up the full height of the viewport */
            }
            [data-testid="stAppViewContainer"] > .main {
                background: none;
                height: 100%;
            }

        </style>""",
    unsafe_allow_html=True)

# Set the width of the displayed images to 300px
st.markdown(
    """<style>
            .st-emotion-cache-1v0mbdj img {
                width: 300px !important;
            }
        </style>""",
    unsafe_allow_html=True)


is_uploaded = False  # variable to control if the file has been uploaded
mask_created = False # variable to control if the mask has been created
if 'auto_generated' not in st.session_state:
    st.session_state['auto_generated'] = False  # variable to check if the mask is automatically generated


# First container
with st.container():
    col11, col12, col13 = st.columns([0.3, 0.5, 0.2], gap='large')
    with col11:
        st.title('Building Segmentation')

    with col12:
        if 'disabled' not in st.session_state:
            st.session_state.disabled = False

        # Image file uploader
        bg_image = st.file_uploader("Choose image to analyze:", type=["png", "jpg"], disabled=st.session_state.disabled,
                                    on_change=enable)

    with col13:
        if bg_image is not None:
            st.session_state.disabled = True
            is_uploaded = True  # variable to control that the image has been uploaded

            # Resizing the uploaded image
            size_up = 300
            img = Image.open(bg_image).resize((size_up, size_up))

            # Selectbox
            drawing_mode = st.selectbox(
                "Segmentation mode:", ("polygon", "automatic"),
                help="- polygon: click to add vertices, right click to close the polygon\n- automatic: select region automatically"
            )

            # Container for post-process button
            button_container = st.container()


if is_uploaded:
    with st.container():
        col21, col22, col23 = st.columns(3)

        with col21:
            st.markdown('<p style="font-size: 20px;">Your Image:</p>',
                        unsafe_allow_html=True)
            if drawing_mode != "automatic":
                st.session_state.auto_generated = False
                # Create a canvas to draw on
                width, height = img.size

                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=3,
                    stroke_color="#E5FB11",
                    background_image=img if bg_image else None,
                    update_streamlit=True,
                    width=width,
                    height=height,
                    drawing_mode=drawing_mode,
                    key="canvas",
                )
                image_data = canvas_result.image_data
            else:
                image_data = None
                st.image(img)  # display loaded image
                if not st.session_state.auto_generated:
                    # Load the segmentation model and perform automatic segmentation
                    unet = load_model(fp="final_model.pth")

                    input_img = preprocess_img(img)

                    mask = predict_mask(unet, input_img)
                    st.session_state["mask"] = mask
                else:
                    mask = st.session_state.mask

                st.session_state.auto_generated = True

        with col22:
            st.markdown('<p style="font-size: 20px;">The selected region will appear here:</p>',
                        unsafe_allow_html=True)

            if image_data is not None or st.session_state.auto_generated:
                # If not automatically generated, create the mask from the image_data
                if image_data is not None and not st.session_state.auto_generated:
                    mask = image_data[:, :, -1] > 0

                if mask.sum() > 0:
                    mask_created = True

                    if st.session_state.auto_generated:
                        # Upscale the mask to display
                        mask_todisp = Image.fromarray(mask).resize((size_up, size_up),
                                                  resample=Image.Resampling.LANCZOS)

                    else:
                        # If not auto-generated simply display the original mask
                        mask_todisp = Image.fromarray(mask)

                    with button_container:
                        # Change button style
                        st.markdown("""<style>
                        
                                    .st-emotion-cache-7ym5gk {
                                        background-color: rgb(145 147 255 / 55%);
                                        border: 2px solid rgb(0 0 0 / 36%);
                                    }
                                    </style>""",
                                    unsafe_allow_html=True)
                        if st.button("Post-process"):
                            # Apply post-processing transformations
                            mask_todisp = post_process_mask(mask)
                            mask_todisp = Image.fromarray(mask_todisp).resize((size_up, size_up),
                                                                          resample=Image.Resampling.LANCZOS)
                    # Display the mask
                    st.image(mask_todisp)

                    with col23:
                        st.markdown('<p style="font-size: 20px;">Detected buildings on your image:</p>',
                                    unsafe_allow_html=True)

                        fig, ax = plt.subplots()
                        ax.imshow(img)
                        ax.imshow(mask_todisp, cmap='jet', alpha=0.3)
                        ax.axis("off")

                        st.pyplot(fig)
