from mode import *
import argparse
import onnx
import matplotlib.pyplot as plt
import onnxruntime as ort
#import optuna
#import optuna.visualization as vis
parser = argparse.ArgumentParser()

from prettytable import PrettyTable

from srgan_model import Generator  # Import the model class definition

######## Optuna for Hyper parameter Optimization.
 def objective(trial):
     # Define search space for hyperparameters
     lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
     vgg_rescale_coeff = trial.suggest_float('vgg_rescale_coeff', 0.001, 0.008)
     adv_coeff = trial.suggest_float('adv_coeff', 0.0001, 0.0015)
     tv_loss_coeff = trial.suggest_float('tv_loss_coeff', 0.001, 0.1)
     batch_size = trial.suggest_int('batch_size', 4, 16)
     args.lr = lr
    #args.pre_train_epoch = pre_train_epoch
    #args.fine_train_epoch = fine_train_epoch
     args.vgg_rescale_coeff = vgg_rescale_coeff
     args.adv_coeff = adv_coeff
     args.tv_loss_coeff = tv_loss_coeff
     args.batch_size = batch_size
     # Run training with the current set of hyperparameters
     val_metric = train(args)  # Modify train() to return the validation metric you want to optimize
     
     return val_metric

def optimize_hyperparameters(args):
    #self.args =args
    study_name = 'image_resolution4'
    storage_name = f'sqlite:///{study_name}.db'
    study = optuna.create_study(study_name=study_name,storage=storage_name,load_if_exists=True,direction='maximize')
    objective = partial(objective_fct, args = args)
    study.optimize(objective, n_trials=3)  # Adjust the number of trials as needed

    # Print the best hyperparameters and the corresponding best value
    print('Best trial:')
    trial = study.best_trial
    print('Value: {}'.format(trial.value))
    print('Params: ')
    for key, value in trial.params.items():
        print('{}: {}'.format(key, value))
    # vis.plot_optimization_history(study)
    # plt.savefig('optimization_history.png')
    loaded_study = optuna.load_study(
            study_name=study_name,
            storage=storage_name,
    )
    print(loaded_study.best_params)
    # vis.plot_optimization_history(study)
    #plot = plot_optimization_history(study)
    # plt.savefig('optimization_history.png')
    #plot.show()
    plot2 = vis.plot_contour(loaded_study)
    plot2.show()



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def str2bool(v):
    return v.lower() in ('true')

parser.add_argument("--LR_path", type = str, default = './test_data')
parser.add_argument("--GT_path", type = str, default = './custom_dataset/train_HR')
parser.add_argument("--res_num", type = int, default = 16)
parser.add_argument("--num_workers", type = int, default = 0)
parser.add_argument("--batch_size", type = int, default = 16)
parser.add_argument("--L2_coeff", type = float, default = 1.0)
parser.add_argument("--adv_coeff", type = float, default = 1e-3)
parser.add_argument("--tv_loss_coeff", type = float, default = 0.0)
parser.add_argument("--pre_train_epoch", type = int, default = 1)
parser.add_argument("--fine_train_epoch", type = int, default = 1)
parser.add_argument("--scale", type = int, default = 4)
parser.add_argument("--patch_size", type = int, default = 24)
parser.add_argument("--feat_layer", type = str, default = 'relu5_4')
parser.add_argument("--vgg_rescale_coeff", type = float, default = 0.006)
parser.add_argument("--fine_tuning", type = str2bool, default = False)
parser.add_argument("--in_memory", type = str2bool, default = True)
parser.add_argument("--generator_path", type = str ,default = './model/without_BN/SRGAN_gene_200.pt')
parser.add_argument("--mode", type = str, default = 'test_only')

args = parser.parse_args()
#start_time = time.time()
if args.mode == 'train':
    #count_parameters()
    #optimize_hyperparameters(args)
    train(args)
    
elif args.mode == 'test':
    test(args)
    
elif args.mode == 'test_only':
    test_only(args)


