import ipdb

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, sizes, dataset='mnist', args=None):

        net_list = []
        if "mnist" in dataset:
            if model_type=="linear":
                for i in range(0, len(sizes) - 1):
                    net_list.append(('linear', [sizes[i+1], sizes[i]], ''))
                    if i < (len(sizes) - 2):
                        net_list.append(('relu', [True], ''))
                    if i == (len(sizes) - 2):
                        net_list.append(('rep', [], ''))
                return net_list

        elif dataset == "tinyimagenet":

            if model_type == 'pc_cnn':
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [640, 16 * channels], ''),
                    ('relu', [True], ''),

                    ('linear', [640, 640], ''),
                    ('relu', [True], ''),
                    ('linear', [sizes[-1], 640], '')
                ]

        elif dataset == "cifar100":


            if model_type == 'pc_cnn':
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),
                    
                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [320, 16 * channels], ''),
                    ('relu', [True], ''),

                    ('linear', [320, 320], ''),
                    ('relu', [True], ''),
                    ('linear', [sizes[-1], 320], '')
                ]

        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)



 