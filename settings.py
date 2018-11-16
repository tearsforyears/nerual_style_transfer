# coding=utf-8

ROOT_DIR = r'F:/resourcefile/vgg19_neural_style_transfer'
PARAMETERS_PATH = ROOT_DIR + r'/data/imagenet-vgg-verydeep-19.mat'

# tensorboard data
SUMMARY_DATA_PATH = ROOT_DIR + r'/visual_data/'

# neural style transfer
# if it has a nice view the parameters had save in comment
CONTENT_IMAGE_PATH = ROOT_DIR + r'/data/tianjin.jpg'
STYLE_IMAGE_PATH = ROOT_DIR + r'/data/van.jpg'
STEPS = 1000  # 500 300
LEARNING_RATE = 50.  # 5. ,50
# two hyper parameters of the loss function to control the style and the
ALPHA = 20  # 500,500*0.04
BETA = 1 * 0.25  # 1,0.25
NOISE_RATIO = .5  # .5,.5
LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
)
# conv1-4_1
STYLE_LAYERS = [
    'relu1_1',
    'relu3_1',
]
CONTENT_LAYER = [
    'conv1_1',
]
TRANSFER_STYLE_IMAGE_SAVE = ROOT_DIR + r'/visual_data'
