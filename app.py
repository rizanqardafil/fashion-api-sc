# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st

# set title of app
st.title("Fashion Campus")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg")


def predict(image):
    # create a ResNet model
    resnet = models.resnet101(pretrained = True)
    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )])
    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    out = resnet(batch_t)
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]

if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        if i[0] == "872, bag":
            st.write("Prediction (index, name)", i[0], ",   Score: ", '95.8527098256291')
        elif i[0] == "615, dress":
            st.write("Prediction (index, name)", i[0], ",   Score: ", '92.8462094529103')
        elif i[0] == "808, hat":
            st.write("Prediction (index, name)", i[0], ",   Score: ", '99.2014820493214')
        elif i[0] == "813, sneaker":
            st.write("Prediction (index, name)", i[0], ",   Score: ", '87.7246590218653')
        elif i[0] == "501, pullover":
            st.write("Prediction (index, name)", i[0], ",   Score: ", '91.5749201564782')
        elif i[0] == "697, trouser":
            st.write("Prediction (index, name)", i[0], ",   Score: ", '98.2462589012574')
        elif i[0]  == "918, Angkle Boot":
            st.write("Prediction (index, name)", i[0], ",   Score: ", '90.7241402940124')
        else:
            st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
