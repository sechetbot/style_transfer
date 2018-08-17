
#https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398

import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
import time

#NB make it so the initial image doesn't have to be the content image
#NB do preprocessing for now but see if you can get it to just take raw unprocessed images


def get_img(name):
    image = cv2.imread('images/' + name).astype(np.float32)
    # image = image/255

    # #my shitty preprocessing
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image /= 127.5
    # image -= 1
    # #NB does the preprocessing even work? if so, go through and postprocess where needed

    #NB new preprocessing, remember to postprocess
    imagenet_mean = np.asarray([103.939, 116.779, 123.68])
    image = image - imagenet_mean


    #NB change the number of pixels to resize to once everything is working, 150 is for speed reasons
    resized = cv2.resize(image, (200,200), interpolation = cv2.INTER_AREA)

    return resized


def get_intermediate_layers():
    #uses the VGG19 network, returns intermediate layers that 'understand' content and style

    #NB make sure input shape is the same as the preprocessed images
    VGG19 = tf.keras.applications.vgg19.VGG19(include_top=False, input_shape=(200,200,3), weights='imagenet')
    VGG19.trainable = False

    style = [VGG19.get_layer(layer).output for layer in style_layers]
    content = [VGG19.get_layer(layer).output for layer in content_layers]

    output = style + content
    #define a model that takes regular inputs and outputs style and content layers
    #NB I orginially wanted to define it to have output style and content seperately, but this doesn't seem to be working in eager
    model = tf.keras.models.Model(inputs=VGG19.input, outputs=output)
    return model


def get_content_loss(input, target):
    #euclidean distance is used for content loss
    return tf.reduce_mean(tf.square(input - target))

def gram_matrix(input_tensor):
    #finds the gram matrix with itself (matrix of all possible dot products)
    channels = input_tensor.shape[-1]
    reshaped = tf.reshape(input_tensor, [-1, channels])
    n = reshaped.shape[0].value
    gram = tf.matmul(reshaped, reshaped, transpose_a=True)
    return gram/n

def get_style_loss(input, target):
    #NB why is gram matrix indicative of style?
    base_gram = gram_matrix(input)
    target_gram = gram_matrix(target)
    return tf.reduce_mean((base_gram - target_gram)**2)

def get_variation_loss(img):
    x_diff = img - tf.manip.roll(img, -1, axis=1)
    y_diff = img - tf.manip.roll(img, -1, axis=0)
    l2 = tf.square(x_diff) + tf.square(y_diff)
    l1 = tf.reduce_sum(tf.sqrt(l2))
    return l1


def get_features(model, content_img, style_img):

    stacked = tf.concat([content_img, style_img], axis=0)
    output = model(stacked)


    #NB look at this indexing and what it means
    style_features = [layer[0] for layer in output[:n_style_layers]]
    content_features = [layer[1] for layer in output[n_style_layers:]]
    return style_features, content_features

def get_loss(model, loss_weights, init_img, style_features, content_features):

    content_weight, style_weight, variation_weight = loss_weights
    gram_style_features = [gram_matrix(feature) for feature in style_features]

    output = model(init_img)
    style_output = output[:n_style_layers]
    content_output = output[n_style_layers:]

    style_loss = 0
    content_loss = 0

    #NB calculate gram_style_features
    #NB look over the style and content loss addition, wtf is happening? what is the form and meaning of content_features???

    style_layer_weight = 1.0/float(n_style_layers)
    for target_style, style_feature in zip(gram_style_features, style_output):
        style_loss += style_layer_weight*get_style_loss(style_feature[0], target_style)

    content_layer_weight = 1.0/float(n_content_layers)
    for target_content, content_feature in zip(content_features, content_output):
        content_loss += content_layer_weight*get_content_loss(content_feature[0], target_content)

    style_loss *= style_weight
    content_loss *= content_weight
    variation_loss = variation_weight * get_variation_loss(init_img)

    total_loss = style_loss + content_loss + variation_loss
    all_loss = (total_loss, style_loss, content_loss, variation_loss)
    return all_loss


def get_gradient(model, loss_weights, init_img, style_features, content_features):
    with tf.GradientTape() as tape:
        tape.watch(init_img)
        all_loss = get_loss(model, loss_weights, init_img, style_features, content_features)
        total_loss = all_loss[0]
    gradient = tape.gradient(total_loss, init_img)

    #NB what is the best replacement value for the nans?
    #find nans and replace them with 0
    gradient = np.nan_to_num(gradient) #NB consider replacing nans with the mean of the array of the array if replacing with 0 propagates the 0 through giving bad results

    return gradient, all_loss

def transfer_style(content_img, style_img, n_iter=1500, loss_weights=(7.5, 100, 10), display_num=10):
    model = get_intermediate_layers()

    style_features, content_features = get_features(model, content_img, style_img)

    #NB change it to be able to predefine and tweak optimiser outside of this
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-1)

    best_loss = float('inf')
    best_img = None

    #showing pictures
    plt.figure(figsize=(15,12))
    n_rows = np.ceil(n_iter/(display_num*5))
    start_time = time.time()
    global_start = time.time()

    for i in range(n_iter):
        gradient, all_loss = get_gradient(model, loss_weights, content_img, style_features, content_features)
        total_loss, style_loss, content_loss, variation_loss = all_loss[0], all_loss[1], all_loss[2], all_loss[3]

        optimizer.apply_gradients([(gradient, content_img)])
        # content_img = tf.clip_by_value(content_img, -0.7, 0.7) #NB this is a temporary fix, the pixels run away to 0 and 1, consider clipping to a mean range
        content_img = tf.contrib.eager.Variable(content_img, dtype=tf.float32)

        end_time = time.time()

        #NB append loss and graph it
        if total_loss < best_loss:
            best_loss = total_loss
            best_img = content_img

        if i % display_num == 0:
            print('Iteration: {}'.format(i + 1))
            print('Loss: {}'.format(total_loss))
            print('Style Loss: {}'.format(style_loss))
            print('Content Loss: {}'.format(content_loss))
            print('TV loss: {}'.format(variation_loss))
            print('Time: {}'.format(end_time - start_time))
            print()

            plt.subplot(n_rows, 5, (i/display_num)+1)
            plottable = tf.squeeze(content_img, axis=0)
            plottable = plottable.numpy() + 1
            plottable *= 0.5
            plt.imshow(plottable)
            plt.title('Iteration: {}'.format(i))

            start_time = time.time()

    plt.show()
    print('Total Time: {}'.format(time.time() - global_start))

    return best_img, best_loss


################################################################################

#we use eager execution to to carry out operations as they are called
tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

content_name = 'Green_Sea_Turtle_grazing_seagrass.jpg'
content_img = get_img(content_name)

style_name = 'The_Great_Wave.jpg'
style_img = get_img(style_name)

imagenet_mean = np.asarray([103.939, 116.779, 123.68])

#preview content and style images
plt.subplot(121)
plt.imshow((content_img + imagenet_mean).astype(int))
plt.subplot(122)
plt.imshow((style_img + imagenet_mean).astype(int))
plt.show()

#need to make the images the right form for inputting into conv2D i.e. (batch, rows, columns, channels)
#content_img = content_img[np.newaxis, :] this form also works
content_img = tf.stack([content_img])
style_img = tf.stack([style_img])

content_img = tf.contrib.eager.Variable(content_img, dtype=tf.float32)
style_img = tf.contrib.eager.Variable(style_img, dtype=tf.float32) #check if it's necessary to make style image a variable, I don't think so

################################################################################
#running it

#content layer used
content_layers = ['block5_conv2']

#style layers used
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']


n_content_layers = len(content_layers)
n_style_layers = len(style_layers)

weights=(1e10, 1, 1)

best_img, best_loss = transfer_style(content_img, style_img, loss_weights=weights, n_iter=400, display_num=50)
best_img = tf.squeeze(best_img, axis=0)

# #my old shitty postprocessing
# best_img = best_img.numpy()+1
# best_img *= 0.5


#NB new postprocessing, remember preprocessing
imagenet_mean = np.asarray([103.939, 116.779, 123.68])
best_img = best_img + imagenet_mean
best_img = best_img.numpy().astype(int)


# best_img = tf.clip_by_value(best_img, 0, 1)

#display and save best image
plt.title('Loss: {}'.format(best_loss))
plt.imshow(best_img)
plt.show()

#cv2 writes in BGR so image colours need to be flipped
# best_img = cv2.cvtColor(best_img.numpy(), cv2.COLOR_BGR2RGB)

#have to multiply by 255 to get correct range for imwrite
# best_img = best_img*255
cv2.imwrite('5 1e10 p.jpg', best_img) #temporary until I get the whole thing fixed, then go back to the other one
# cv2.imwrite('created/CSV loss: {}.jpg'.format(weights), best_img.numpy()) #temporary until I get the whole thing fixed, then go back to the other one
# cv2.imwrite('created/{} with {}.jpg'.format(content_name, style_name), best_img)
