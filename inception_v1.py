import torch
import torch.nn as nn
import torch.nn.functional as F
import os

'''

This module is a pytorch translation of https://github.com/soumith/inception.torch 

'''

def layer_init(m):
    classname = m.__class__.__name__
    classname = classname.lower()
    if classname.find('conv') != -1 or classname.find('linear') != -1:
        gain = nn.init.calculate_gain(classname)
        nn.init.xavier_uniform(m.weight, gain=gain)
        nn.init.constant(m.bias, 0)
    elif classname.find('batchnorm') != -1:
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)

class LRN(nn.Module):

    '''
    Implementing Local Response Normalization layer. Implemention adapted from
    https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py
    '''

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class Inception_base(nn.Module):
    def __init__(self, depth_dim, input_size, config):
        super(Inception_base, self).__init__()

        self.depth_dim = depth_dim

        #mixed 'name'_1x1
        self.conv1 = nn.Conv2d(input_size, out_channels=config[0][0], kernel_size=1, stride=1, padding=0)

        #mixed 'name'_3x3_bottleneck
        self.conv3_1 = nn.Conv2d(input_size, out_channels=config[1][0], kernel_size=1, stride=1, padding=0)
        #mixed 'name'_3x3
        self.conv3_3 = nn.Conv2d(config[1][0], config[1][1], kernel_size=3, stride=1, padding=1)

        # mixed 'name'_5x5_bottleneck
        self.conv5_1 = nn.Conv2d(input_size, out_channels=config[2][0], kernel_size=1, stride=1, padding=0)
        # mixed 'name'_5x5
        self.conv5_5 = nn.Conv2d(config[2][0], config[2][1], kernel_size=5, stride=1, padding=2)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1)
        #mixed 'name'_pool_reduce
        self.conv_max_1 = nn.Conv2d(input_size, out_channels=config[3][1], kernel_size=1, stride=1, padding=0)

    def forward(self, input):

        output1 = F.relu(self.conv1(input))

        output2 = F.relu(self.conv3_1(input))
        output2 = F.relu(self.conv3_3(output2))

        output3 = F.relu(self.conv5_1(input))
        output3 = F.relu(self.conv5_5(output3))

        output4 = F.relu(self.conv_max_1(self.max_pool_1(input)))

        return torch.cat([output1, output2, output3, output4], dim=self.depth_dim)


class Inception_v1(nn.Module):
    def __init__(self, num_classes=1000):
        super(Inception_v1, self).__init__()

        #conv2d0
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn1 = LRN(local_size=11, alpha=0.00109999999404, beta=0.5, k=2)

        #conv2d1
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        #conv2d2
        self.conv3  = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.lrn3 = LRN(local_size=11, alpha=0.00109999999404, beta=0.5, k=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_3a = Inception_base(1, 192, [[64], [96,128], [16, 32], [3, 32]]) #3a
        self.inception_3b = Inception_base(1, 256, [[128], [128,192], [32, 96], [3, 64]]) #3b
        self.max_pool_inc3= nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.inception_4a = Inception_base(1, 480, [[192], [ 96,204], [16, 48], [3, 64]]) #4a
        self.inception_4b = Inception_base(1, 508, [[160], [112,224], [24, 64], [3, 64]]) #4b
        self.inception_4c = Inception_base(1, 512, [[128], [128,256], [24, 64], [3, 64]]) #4c
        self.inception_4d = Inception_base(1, 512, [[112], [144,288], [32, 64], [3, 64]]) #4d
        self.inception_4e = Inception_base(1, 528, [[256], [160,320], [32,128], [3,128]]) #4e
        self.max_pool_inc4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_5a = Inception_base(1, 832, [[256], [160,320], [48,128], [3,128]]) #5a
        self.inception_5b = Inception_base(1, 832, [[384], [192,384], [48,128], [3,128]]) #5b
        self.avg_pool5 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.dropout_layer = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        self.apply(layer_init)

    def forward(self, input):

        output = self.max_pool1(F.relu(self.conv1(input)))
        output = self.lrn1(output)

        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = self.max_pool3(self.lrn3(output))

        output = self.inception_3a(output)
        output = self.inception_3b(output)
        output = self.max_pool_inc3(output)

        output = self.inception_4a(output)
        output = self.inception_4b(output)
        output = self.inception_4c(output)
        output = self.inception_4d(output)
        output = self.inception_4e(output)
        output = self.max_pool_inc4(output)

        output = self.inception_5a(output)
        output = self.inception_5b(output)
        output = self.avg_pool5(output)

        output = output.view(-1, 1024)
        output = self.dropout_layer(output)
        output = self.fc(output)

        return output


def inception_v1_pretrained(path_to_weights='default'):
    model = Inception_v1(num_classes=1000)
    if path_to_weights.lower() == 'default':
        _currDir = os.path.dirname(os.path.abspath(__file__))
        path_to_weights = os.path.join(_currDir, 'inception_v1_weights')
    model.load_state_dict(torch.load(path_to_weights))
    return model


dictionary_default_pytorch_names_to_correct_names_full = {
    'conv1':'conv2d0',
    'conv2':'conv2d1',
    'conv3':'conv2d2',
    'fc':'softmax2'
}

dictionary_default_pytorch_names_to_correct_names_base = {
    'conv1':   'mixed{}_1x1',
    'conv3_1': 'mixed{}_3x3_bottleneck',
    'conv3_3': 'mixed{}_3x3',
    'conv5_1': 'mixed{}_5x5_bottleneck',
    'conv5_5': 'mixed{}_5x5',
    'conv_max_1': 'mixed{}_pool_reduce'
}

def load_weights_from_dump(model, dump_folder):

    'Loads the weights saved as h5py files in the soumith repo linked above. Just here for completeness'

    dump_folder = os.path.abspath(dump_folder)
    for name, layer in model.named_parameters():
        #get path from name
        if 'inception' in name:
            first_dot = name.find('.')
            name_inception = name[:first_dot].replace('inception_', '')
            name_layer = name[first_dot+1:name.find('.', first_dot+1)]
            name_layer = dictionary_default_pytorch_names_to_correct_names_base[name_layer].format(name_inception)
        else:
            name_layer = name[:name.find('.')]
            name_layer = dictionary_default_pytorch_names_to_correct_names_full[name_layer]
        if 'weight' in name:
            filename = name_layer + '_w.h5'
        else:
            filename = name_layer + '_b.h5'

        filename = os.path.join(dump_folder, filename)
        #print(filename, 'exists', os.path.isfile(filename))

        f = h5py.File(filename, 'r')
        a_group_key = list(f.keys())[0]
        w = np.asarray(list(f[a_group_key]))

        w = torch.from_numpy(w)
        if 'weight' in name:
            w = w.transpose(1, 3).transpose(2, 3)
        if name_layer == 'softmax2':
            w = w.squeeze()
            # Adjust the size - because google has 1008 classes, class 1 - 1000 are valid
            if 'weight' in name:
                w = w[1:1001, :]
            else:
                w = w[1:1001]

        if layer.data.size() != w.size():
            raise ValueError('Incompatible sizes')
        layer.data = w

    return model
