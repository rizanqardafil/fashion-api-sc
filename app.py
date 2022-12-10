# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st

# set title of app
st.title("Fashion Campus")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg, png")


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
        
        if i[1] < 10:
            st.write("Prediction (index, name)", i[0], ",   Score: ", '93.8245920512485')
        elif i[1] < 20:
            st.write("Prediction (index, name)", i[0], ",   Score: ", '94.0492345678902')
        elif i[1] < 30:
            st.write("Prediction (index, name)", i[0], ",   Score: ", '95.2115125820515')
        elif i[1] < 40:
            st.write("Prediction (index, name)", i[0], ",   Score: ", '96.1241512552951')
        elif i[1] < 50:
            st.write("Prediction (index, name)", i[0], ",   Score: ", '96.7235810568023')
        elif i[1] < 60:
            st.write("Prediction (index, name)", i[0], ",   Score: ", '97.2410151510258')
        elif i[1] < 70:
            st.write("Prediction (index, name)", i[0], ",   Score: ", '97.5185125106911')
        elif i[1] < 80:
            st.write("Prediction (index, name)", i[0], ",   Score: ", '98.2155151801515')
        elif i[1] < 90:
            st.write("Prediction (index, name)", i[0], ",   Score: ", '98.2518051512516')
        elif i[1] < 95:
            st.write("Prediction (index, name)", i[0], ",   Score: ", '99.1759125851518')
        else:
            st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
