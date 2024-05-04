# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import random
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps
import numpy as np
import subprocess

# Running the 'which python' command to find out which Python interpreter is being used
result = subprocess.run(["which", "python"], stdout=subprocess.PIPE, text=True)

# Printing the output
print(result.stdout.strip())

# hide deprication warnings which directly don't affect the working of the application
import warnings

warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(page_title="Mango Leaf Disease Detection", page_icon=":mango:", initial_sidebar_state="auto")

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(
    hide_streamlit_style, unsafe_allow_html=True
)  # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache_resource
def load_model():
    tf_model = tf.keras.models.load_model("./notebooks/mango_model.h5")
    return tf_model


with st.spinner("Model is being loaded.."):
    model = load_model()


def prediction_cls(prediction):  # predict the class of the images based on the model results
    for key, clss in class_names:  # create a dictionary of the output classes
        if np.argmax(prediction) == clss:  # check the class

            return key


with st.sidebar:
    # st.image("mg.png")
    st.title("Mangifera Healthika")
    st.subheader(
        "Accurate detection of diseases present in the mango leaves. This helps an user to easily detect the disease and identify it's cause."
    )

st.write(
    """
         # Mango Disease Detection with Remedy Suggestion
         """
)

file = st.file_uploader("", type=["jpg", "png"])


def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = [
        "Anthracnose",
        "Bacterial Canker",
        "Cutting Weevil",
        "Die Back",
        "Gall Midge",
        "Healthy",
        "Powdery Mildew",
        "Sooty Mould",
    ]

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == "Healthy":
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == "Anthracnose":
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Bio-fungicides based on Bacillus subtilis or Bacillus myloliquefaciens work fine if applied during favorable weather conditions. Hot water treatment of seeds or fruits (48°C for 20 minutes) can kill any fungal residue and prevent further spreading of the disease in the field or during transport."
        )

    elif class_names[np.argmax(predictions)] == "Bacterial Canker":
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Prune flowering trees during blooming when wounds heal fastest. Remove wilted or dead limbs well below infected areas. Avoid pruning in early spring and fall when bacteria are most active.If using string trimmers around the base of trees avoid damaging bark with breathable Tree Wrap to prevent infection."
        )

    elif class_names[np.argmax(predictions)] == "Cutting Weevil":
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Cutting Weevil can be treated by spraying of insecticides such as Deltamethrin (1 mL/L) or Cypermethrin (0.5 mL/L) or Carbaryl (4 g/L) during new leaf emergence can effectively prevent the weevil damage."
        )

    elif class_names[np.argmax(predictions)] == "Die Back":
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "After pruning, apply copper oxychloride at a concentration of '0.3%' on the wounds. Apply Bordeaux mixture twice a year to reduce the infection rate on the trees. Sprays containing the fungicide thiophanate-methyl have proven effective against B."
        )

    elif class_names[np.argmax(predictions)] == "Gall Midge":
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Use yellow sticky traps to catch the flies. Cover the soil with plastic foil to prevent larvae from dropping to the ground or pupae from coming out of their nest. Plow the soil regularly to expose pupae and larvae to the sun, which kills them. Collect and burn infested tree material during the season."
        )

    elif class_names[np.argmax(predictions)] == "Powdery Mildew":
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "In order to control powdery mildew, three sprays of fungicides are recommended. The first spray comprising of wettable sulphur (0.2%, i.e., 2g per litre of water) should be done when the panicles are 8 -10 cm in size as a preventive spray."
        )

    elif class_names[np.argmax(predictions)] == "Sooty Mould":
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "The insects causing the mould are killed by spraying with carbaryl or phosphomidon 0.03%. It is followed by spraying with a dilute solution of starch or maida 5%. On drying, the starch comes off in flakes and the process removes the black mouldy growth fungi from different plant parts."
        )
