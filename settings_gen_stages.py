t_name = 'all_generation'
task_kwargs={ 
    'setup': 'multi_task',
    'task_names':['gen1', 'gen2', 'gen3', 'gen4'],
    'task_outchannel':3,
    'task_type':{
        'gen1': 'gen',
        'gen2': 'gen',
        'gen3': 'gen',
        'gen4': 'gen',
    },
    'loss_weights':{
        'gen1': 1.,
        'gen2': 1.,
        'gen3': 1.,
        'gen4': 1.,
    }
}

# binary segment
# t_name = 'gen1' # 'gen1' | 'gen2' | 'gen3' | 'gen4'
# task_kwargs={ 
#     'setup': 'single_task',
#     'task_names':[t_name],
#     'task_outchannel':{
#         t_name: 3,
#     },
#     'task_type':{
#         t_name: 'gen',
#     },
#     'loss_weights':{
#         t_name: 1.,
#     }
# }

# 2048 = 1500*4+548=6000+
# proto_combine_mode = 'proto_as_feature'
prototype_kwargs={
    'num_per_task':10,
    'num_shared':5,
    'shape':(45, 512, 1, 1),
}
proto_combine_mode = 'sum' # concat | sum | concatcascade | proto_as_feature
cluster_weight = 0 # 1e-2 | 0

base_architecture = 'resnet50' # 'vgg19'
decoder_channels=(256, 128, 64, 32, 16) # (1024, 512, 256, 128, 64) | (256, 128, 64, 32, 16) | (512, 256, 128, 64, 32)
img_size = 256
# prototype_shape = (15, 128, 1, 1) # (2000, 128, 1, 1)
# num_classes = 10 # 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '{}_zjj_sgd1e2_dc{}_noloss'.format(proto_combine_mode, decoder_channels[0])
# experiment_run = 'Dice'

# img_scale=1.0
foldk = 9
data_path = '/home/test/zjj/data/work3-multitask/mIHC_adjust'
train_dir = data_path + 'train-val/' # 'train_cropped_augmented/'
test_dir = data_path + 'test/' #  'test_cropped/'
train_push_dir = data_path + 'train-val/' # 'train_cropped/'
train_batch_size = 32 # 16
test_batch_size = 16 # 100
train_push_batch_size = 16 # 16

joint_optimizer_lrs = {'features': 1e-3,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 5e-3,
                       'selayers': 3e-3,
                       'warm_decoder': 3e-3,
                       'decoders': 5e-3}
# joint_optimizer_lrs = {'features': 1e-4,
#                        'add_on_layers': 3e-4,
#                        'prototype_vectors': 3e-4,
#                        'selayers': 3e-4,
#                        'warm_decoder': 3e-4,
#                        'decoders': 3e-4}

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 5e-3,
                      'selayers': 3e-3,
                      'warm_decoder': 3e-3,
                      'decoders': 5e-3}
# warm_optimizer_lrs = {'add_on_layers': 3e-4,
#                       'prototype_vectors': 3e-4,
#                       'selayers': 3e-4,
#                       'warm_decoder': 3e-4,
#                       'decoders': 3e-4}

last_layer_optimizer_lr = 1e-4

coefs = {
    'task': 1,
    'clst': 0,
    'sep': 0,
    'kld': 1e-2,
}

eval_per_epoch = 10

joint_lr_step_size = 100

num_warm_epochs = 0
num_stage1_epochs = 50
num_train_epochs = 1000

# num_warm_epochs = 1
# num_stage1_epochs = 2
# num_train_epochs = 1000

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0 and i != 0]
