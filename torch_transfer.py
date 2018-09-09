
# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import time

################################################################################
#handy small functions
def get_img_as_tensor(name):
    image = cv2.imread('images/' + name).astype(np.float32)

    #torch preprocessing, torch takes float images, values (0, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image /= 255

    #NB change the number of pixels to resize to once everything is working, 250 is for speed reasons
    resized = cv2.resize(image, (500,500), interpolation = cv2.INTER_AREA)
    tensorform = torch.tensor(resized, dtype=torch.float32).unsqueeze(0)
    return tensorform


def display_tensor_img(tensor):
        image = tensor.squeeze(0)
        plt.imshow(image)
        return


def show_start_imgs(img1, img2, option):
    if option:
        plt.subplot(121)
        display_tensor_img(img1)
        plt.subplot(122)
        display_tensor_img(img2)
        plt.show()


def torch_gram(input):
    a, b, c, d = input.size() #this is the form produced by pytorch's VGG19, [batches, feature_maps, map_dimension1, map_dimension2]
    feature_matrix = input.view(a*b, c*d) #this is the matrix we gram
    gram = torch.mm(feature_matrix, feature_matrix.t()) #NB why does the gram matrix represent style?
    return gram.div(a*b*c*d) #normalize to ensure equal weightings for different sized layer outputs


def get_model_and_losses(network, content_img, style_img, norm_mean, norm_std, content_layers, style_layers):

    content_losses = []
    style_losses = []

    normalize = preprocess(norm_mean, norm_std)
    model = nn.Sequential(normalize)

    i = 0
    for layer in network.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False) #the inbuilt VGG19 network uses an inplace relu, which apparently doesn't work well with our content and style losses
        elif isinstance(layer, nn.MaxPool2d):
            name = 'maxpool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'batchnorm_{}'.format(i)
        else:
            raise RuntimeError('Unrecognised layer: {}'.format(layer.__class__.name))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            c_loss = content_loss(target)
            model.add_module('contentloss_{}'.format(i), c_loss)
            content_losses.append(c_loss)

        if name in style_layers:
            target = model(style_img).detach()
            s_loss = style_loss(target)
            model.add_module('styleloss_{}'.format(), s_loss)
            style_losses.append(s_loss)

    #Removing the layers after the last loss layer
    for i in range(len(model), 0, -1):
        print(i)
        if isinstance(model[i], content_loss) or isinstance(model[i], style_loss):
            break

    model = model[:(i+1)]

    return model, content_losses, style_losses


def get_input_optimizer(input_img):
    optimizer = torch.optim.LBFGS([input_img.requires_grad()])
    return optimizer


def style_transfer(network, mean, std, content_img, style_img, input_img, content_layers, style_layers,
    n_iter=500, s_weight=1e7, c_weight=1):

    print("Building the model:")
    model, content_losses, style_losses = get_model_and_losses(network, style_img, content_img, mean, std, content_layers, style_layers)
    optimizer = get_input_optimizer(input_img)

    print("Minimizing the losses:")
    for i in range(n_iter):
        input_img.data.clamp_(0,1)

        optimizer.zero_grad()
        model(input_img)
        style_loss = 0
        content_loss = 0

        for cl in content_losses:
            content_loss += cl
        for sl in style_losses:
            style_loss += sl

        content_loss *= c_weight
        style_loss *= s_weight

        loss = content_loss + style_loss
        loss.backward()

        if i%50 == 0:
            print('iteration number: {}'.format(i))
            print('content loss: {}, style loss: {}'.format(content_loss.item(), style_loss.item()))
            print()

        optimizer.step()

    input_img.data.clamp_(0, 1)
    return input_img
################################################################################
#making useful modules
class preprocess(nn.Module):
    def __init__(self, mean, std):
        super(preprocess, self).__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1) #making sure the mean and std are the right shape for our tensors
    def forward(input_img):
        return (input_img - self.mean)/self.std

class content_loss(nn.Module):
    def __init__(self, target):
        super(content_loss, self).__init__()
        self.target = target.detach() #You have to detach the target, as it isn't a variable but a constant NB look into why

    def forward(self, input):
        self.c_loss = nn.functional.mse_loss(input, self.target)
        return input

class style_loss(nn.Module):
    def __init__(self, target_layer_output):
        super(style_loss, self).__init__
        self.target_gram = torch_gram(target_layer_output).detach() #like before it must be detached, I am unsure why

    def forward(self, input):
        self.s_loss = nn.funtional.mse_loss(torch_gram(input), self.target_gram)
        return input
################################################################################
#actually running it
style_img = get_img_as_tensor('The_Great_Wave.jpg')
content_img = get_img_as_tensor('Green_Sea_Turtle_grazing_seagrass.jpg')

# show_start_imgs(content_img, style_img, True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(device))

#this line for content image initilaization
# input_img = content_img

#this line is for white noise initialization
input_img = torch.randn(content_img.data.size(), device=device)


network = torchvision.models.vgg19(pretrained=True).features.to(device).eval() #to(device) determines whether we use gpu or cpu, .eval() means it is not trainable

normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device) #why the to(device) here? NB remove it and see

#style and content layers to use
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

output = style_transfer(network, normalization_mean, normalization_std,
                            content_img, style_img, input_img,
                            content_layers_default, style_layers_default)

plt.figure()
imshow(output, title='Output Image')
