
Args = {"name" : 'model_f_Adam_lrMax0.001_batch64',
        "lr" : 0,
        #"weight_decay" : 0.001,
        "batch_size": 64, # 2 gpus 80, 1 gpu 32
        "Epoch" : 200,
        "patience" : 30,
        #"device" : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        }

