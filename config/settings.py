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

prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001'

# img_scale=1.0
datasetname = 'liver'
data_path = 'dataset'
train_dir = data_path + 'train/' # 'train_cropped_augmented/'
val_dir = data_path + 'val/'
test_dir = data_path + 'test/' #  'test_cropped/'
train_push_dir = data_path + 'train/' # 'train_cropped/'
train_batch_size = 32 # 16
test_batch_size = 16 # 100
train_push_batch_size = 16 # 16

coefs = {
    'task': 1,
    'act': 0.1,
    'div': 0.1,
}

eval_per_epoch = 10 

joint_lr_step_size = 100

num_stage1_epochs = 15
num_train_epochs = 1001

push_start = 10 
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0 and i != 0]
