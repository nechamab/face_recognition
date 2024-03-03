import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns

# st.title('Practice app')
# st.header('gonna make an app good')

# # put contact info in here
# st.sidebar.title('sidebar title')
# sidebar = st.sidebar 

# # button
# side_button = st.sidebar.button('Press me')
# if side_button:
#     sidebar.write('button is pressed')
# else:
#     sidebar.write('Not pressed yet')

# # columns 

# col1, col2 = st.columns(2)
# col1.subheader('Col1')
# col2.subheader('Col2')

# col21, col22, col23 = st.columns([3, 2, 1])
# col21.write('this is the widest column and the text should wrap appropriately')
# col22.write('second to widest, kinda mid, basic, testing if thats true tbh')
# col23.write('this is smallest check ok cool')

# st.markdown('**yoooo**')

# '## markdown in reg space'

# # st.write('<h2 style='text-align:center'')

# check = st.checkbox('check this out')
# button_check = st.button('is box chekced')
# if button_check:
#     if check:
#         st.write('box is checked')
#     else:
#         st.write('box is not checked')

# input_options = ['Upload an image', 'Use Camera']
# option = st.radio('Choose an option', input_options)
# submit_button = st.button('submit')
# if submit_button:
#     st.write(f'you chose {option} cool')


# select_box = st.selectbox('Choose an option', input_options)
# submit_button = st.button('submit2')
# if submit_button:
#     st.write(f'you chose {select_box} cool')

# num_stuff = st.slider('how many things do you have', 0, 10, step=1)

# in_text = st.text_input('what is your fav animal', value = 'i dont know')

# # penguins = sns.load_dataset('penguins')

# # data_expander = st.expander('Show penguins dataset')
# # data_expander.table(penguins.head())

import cv2
import streamlit as st
import cv2
import os
import time

# from predict import Classify

# Set paths of directories

INPUT_PATH = os.path.join('application_data', 'input_image')


def main():
    """Face Recognition App"""


    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition Web Application</h2>
    </div>
    </body>
    <body style="background-color:red;">
    <div >
    <h3 style="color:white;text-align:center;"></h3>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)



    run = st.checkbox('**Start Webcam to Capture Face**')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    st.info('Please click the button to capture face') 
    brk= st.button('Capture')
    while run:
        _, frame = camera.read()
        frame = cv2.resize(frame,(250,250))
        
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame1)
        path = os.path.join(INPUT_PATH,"input_image.jpg")
        cv2.imwrite(path,frame)
        if brk:
            st.success("Image captured successfully!!! Please wait for the verification result!!!")
            
            break
        
    #st.write('Verification Result')  
             
    try:
        filename = "application_data\input_image\input_image.jpg"
        classifier = Classify(filename)
        result = classifier.recognition()
        os.remove(filename)
        st.success("Verification Result!!!")
        st.text(result)
    except:
        st.write("************************************************")
        

    


if __name__ == '__main__':
    main()
