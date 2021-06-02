import tensorflow as tf
import numpy as np
import os, subprocess
from deprecated import deprecated
from models.losses import conf_interval, nrmse_batch_np, nd_samples_batch_np

def auto_gpu_selection(usage_max=0.01, mem_max=0.05, is_tensorflow=True):
    """Auto set CUDA_VISIBLE_DEVICES for gpu

    :param mem_max: max percentage of GPU utility
    :param usage_max: max percentage of GPU memory
    :return:
    """
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    log = str(subprocess.check_output("nvidia-smi", shell=True)).split(r"\n")[6:-1]
    gpu = 0

    # Maximum of GPUS, 8 is enough for most
    for i in range(8):
        idx = i*3 + 2
        if idx > log.__len__()-1:
            break
        inf = log[idx].split("|")
        if inf.__len__() < 3:
            break
        usage = int(inf[3].split("%")[0].strip())
        mem_now = int(str(inf[2].split("/")[0]).strip()[:-3])
        mem_all = int(str(inf[2].split("/")[1]).strip()[:-3])
        # print("GPU-%d : Usage:[%d%%]" % (gpu, usage))
        if usage < 100*usage_max and mem_now < mem_max*mem_all:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            if is_tensorflow:
                # Set memory growth
                gpu_devices = tf.config.experimental.list_physical_devices('GPU')
                for device in gpu_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            print("\nAuto choosing vacant GPU-%d : Memory:[%dMiB/%dMiB] , GPU-Util:[%d%%]\n" %
                (gpu, mem_now, mem_all, usage))
            return
        print("GPU-%d is busy: Memory:[%dMiB/%dMiB] , GPU-Util:[%d%%]" %
            (gpu, mem_now, mem_all, usage))
        gpu += 1
    print("\nNo vacant GPU, use CPU instead\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def use_gpu(USE_GPU: bool, GPU_ID=1):
    if USE_GPU:
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print("Exception occured, training on CPU !!!")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
def log_errors(dataset, model, wandb, prefix='best-', save_residuals=True):
    """ Log the errors to wandb
    """
    # Predict
    y_pred = model.predict(dataset['test_x'])
    y_true = dataset['test_y']
    # NOTE: Descale (Electricity and Rossmann don't have scaler)
    if type(dataset['target_scaler']) != dict and dataset['target_scaler'] is not None:
        y_pred = dataset['target_scaler'].inverse_transform(y_pred.reshape(-1,1)).reshape(y_pred.shape)
        y_true = dataset['target_scaler'].inverse_transform(y_true.reshape(-1,1)).reshape(y_true.shape)
    else:
        print('Target scaler not found, reporting errors for scaled values')
    
    if save_residuals:
        np.save(os.path.join(wandb.run.dir, 'residuals.npy'), y_true-y_pred)

    nrmse_maxmin = nrmse_batch_np(y_true, y_pred, divide_by='max-min', batch_size=256)
    nrmse_mean = nrmse_batch_np(y_true, y_pred, divide_by='mean', batch_size=256)
    nd = nd_samples_batch_np(y_true, y_pred, batch_size=256)

    wandb.log({
        prefix+'nrmse_maxmin': nrmse_maxmin[0],
        prefix+'nrmse_maxmin_std': nrmse_maxmin[1],
        prefix+'nrmse_maxmin_95': nrmse_maxmin[2],

        prefix + "nd" : nd[0],
        prefix + "nd_std": nd[1],
        prefix + "nd_95": nd[2],

        prefix+'nrmse_mean': nrmse_mean[0],
        prefix+'nrmse_mean_std': nrmse_mean[1],
        prefix+'nrmse_mean_95': nrmse_mean[2],
    })
    
