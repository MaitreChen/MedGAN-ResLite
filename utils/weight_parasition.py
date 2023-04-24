import torch
from collections import OrderedDict

model = torch.load("../pretrained/resnet18.pth")
print(model.keys())
a = model['fc.weight'][:2, :]

new_dict = OrderedDict()
for key in model.keys():
    if key == 'conv1.weight':
        input_dict = model[key][:, 0, :, :]
        input_dict_ = torch.unsqueeze(input_dict, dim=1)
        new_dict[key] = input_dict_
    elif key == 'fc.weight':
        fc_weight = model[key][:2, :]
        new_dict[key] = fc_weight
    elif key == 'fc.bias':
        fc_bias = model[key][:2]
        new_dict[key] = fc_bias
    else:
        continue

print(new_dict['conv1.weight'].shape)
print(new_dict['fc.weight'].shape)
print(new_dict['fc.bias'].shape)

torch.save(new_dict, '../pretrained/new_resnet.pth')
