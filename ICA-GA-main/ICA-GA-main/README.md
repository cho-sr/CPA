# ICA-driven Generative Attacks (ICA-GA)

![](README_ica_ga.png)

This is the code implementation for the paper "**Unmixing Gradients: Uncovering Persistent Leakage in Federated Learning via Independent Component Analysis**".

In this work, we propose ICA-GA, a gradient leakage attack framework driven by the Independent Component Analysis (ICA) algorithm and based on the generative attack paradigm. By reframing the feature separation task as a classic Blind Source Separation problem from the field of signal processing, our method overcomes the limitations of existing approaches and demonstrates a more robust and persistent threat throughout the entire lifecycle of federated learning.

Paper Link: https://doi.org/10.1016/j.sysarc.2025.103681

## 1. Preparation

### 1.1 Get the Source Code

```sh
git clone https://github.com/liminhao-99/ICA-GA.git
cd ICA-GA
```

### 1.2. Create a Conda Environment

```sh
conda create --name ica-ga python=3.12 -y
conda activate ica-ga
pip install -r requirements.txt
```

### 1.3. Download Pre-trained Models

To quickly reproduce the core experiments from the paper, it is highly recommended to download our pre-trained model files.

- `ResNet50_no-avgpool_out-100_cifar100_fine-tuned.pth`: A ResNet-50 model. Fully fine-tuned on Cifar100 based on PyTorch's default pre-trained weights. Used as the FL model.
- `gen_1x_CT_RCSE_r50_fine_tuned.pth`: A trained lightweight generator that matches the FL model above.

```sh
# Create data directories
mkdir -p data/fl_models
mkdir -p data/gen_models
mkdir -p data/temp

# Download models
wget -O data/temp/models_fine-tuned.zip https://github.com/liminhao-99/ICA-GA/releases/download/models/models_fine-tuned.zip

# Unzip the archive
unzip -q data/temp/models_fine-tuned.zip -d data/temp

# Move the models to their final destination
mv data/temp/fl_models/ResNet50_no-avgpool_out-100_cifar100_fine-tuned.pth data/fl_models/ResNet50_no-avgpool_out-100_cifar100_fine-tuned.pth
mv data/temp/gen_models/gen_1x-CT-RCSE_r50-fine-tuned.pth data/gen_models/gen_1x-CT-RCSE_r50-fine-tuned.pth

# rm -r data/temp # Clean up temporary files
```

### 1.4. Prepare Datasets

- **Attack Test Datasets (CIFAR-100, etc.)**: The code will automatically download these to the `./data/datasets` directory when you run an experiment for the first time.
- **Generator Training Auxiliary Dataset (ImageNet)**: This needs to be downloaded manually. See subsequent instructions for details.

## 2. Attack Experiments

You can run different experiments by modifying the command-line arguments.

- **Argument Format**: `-key=value`. Please note that arguments must start with a `-`, use `=` for assignment, and contain no extra spaces.
- **More Arguments**: Each experiment script (e.g., `fedsgd_attacks.py`) contains more detailed argument descriptions in its comments.

The experiment results will be saved to the specified CSV file. For experiments that are repeated multiple times, the `test_num` column in the table typically indicates the experiment number, and a final `all` row records the mean of all metrics.

### 2.1. FedSGD Attack Test

```sh
python3 exps/fedsgd_attacks/fedsgd_attacks.py \
    -result_dir=results\
    -result_name=result.csv\
    -fl_model_load.load_dir=./data/fl_models\
    -fl_model_load.load_name=ResNet50_no-avgpool_out-100_cifar100_fine-tuned\
    -fl_model_load.load_fc=False\
    -fl_model_kwargs.output_length=1000\
    -fl_model_kwargs.is_avgpool=False\
    -attacks.ICA-GA.model_save_dir=./data/gen_models\
    -attacks.ICA-GA.model_save_name=gen_1x-CT-RCSE_r50-fine-tuned\
    -data_iter_kwargs.dataset_name=cifar100\
    -data_iter_kwargs.batch_size=32\
    -device=cuda:0\
    -seed=42\
    -data_iter_kwargs.dataloader_seed=42\
    -test_batchs=10\
    -rep_rate=0.0
```

Tip: Set `-rep_rate=1.0` to create an experimental environment with 100% label repetition.

| Basic Argument                                                         | Description                                                                                              |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| result\_dir=results                                                     | Directory to save results (relative to fedsgd\_attacks.py)                                                |
| result\_name=result.csv                                                 | Output filename for results, as a CSV file                                                               |
| fl\_model\_load.load\_dir=./data/fl\_models                                | Directory to load the FL model from (relative to the project root)                                       |
| fl\_model\_load.load\_name=ResNet50\_no-avgpool\_out-100\_cifar100\_fine-tuned | FL model filename (without the .pth suffix)                                                              |
| fl\_model\_load.load\_fc=False                                            | Do not load the classifier for the FL model (use initialized parameters)                                 |
| fl\_model\_kwargs.output\_length=1000                                     | Specify the number of classes for the FL model                                                           |
| fl\_model\_kwargs.is\_avgpool=False                                       | Remove the average pooling layer from the FL model (required)                                            |
| attacks.ICA-GA.model\_save\_dir=./data/gen\_models                        | Directory to load the generator model from                                                               |
| attacks.ICA-GA.model\_save\_name=gen\_1x-CT-RCSE\_r50-fine-tuned           | Generator model filename (without the .pth suffix)                                                       |
| data\_iter\_kwargs.dataset\_name=cifar100                                 | Test dataset. Options: caltech256, cifar10, cifar100, mnist, imagenet                                    |
| data\_iter\_kwargs.batch\_size=32                                         | Test batch size                                                                                          |
| device=cuda:0                                                          | Device for the experiment                                                                                |
| seed=42                                                                | Main random seed                                                                                         |
| data\_iter\_kwargs.dataloader\_seed=42                                    | Random seed for loading test data                                                                        |
| test\_batchs=10                                                         | Number of times to repeat the experiment                                                                 |
| rep\_rate=0.0                                                           | [Optional] Label repetition rate: If set to 0.0 \< rep\_rate \<= 1.0, constructs special data batches where `rep_rate` of the samples share the same label |

### 2.2. FedAVG Attack Test

```sh
python3 exps/fedavg_attacks/fedavg_attacks.py \
    -result_dir=results\
    -result_name=result.csv\
    -fl_model_load.load_dir=./data/fl_models\
    -fl_model_load.load_name=ResNet50_no-avgpool_out-100_cifar100_fine-tuned\
    -fl_model_load.load_fc=False\
    -fl_model_kwargs.is_avgpool=False\
    -fl_model_kwargs.output_length=1000\
    -attacks.ICA-GA.model_save_dir=./data/gen_models\
    -attacks.ICA-GA.model_save_name=gen_1x-CT-RCSE_r50-fine-tuned\
    -data_iter_kwargs.dataset_name=cifar100\
    -data_iter_kwargs.batch_size=8\
    -data_iter_kwargs.local_steps=4\
    -data_iter_kwargs.local_epochs=1\
    -seed=42\
    -data_iter_kwargs.dataloader_seed=42\
    -test_batchs=10\
    -device=cuda:0
```

| Basic Argument                                                         | Description                                                                 |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| result\_dir=results                                                     | Directory to save results (relative to fedavg\_attacks.py)                   |
| result\_name=result.csv                                                 | Output filename for results, as a CSV file                                  |
| fl\_model\_load.load\_dir=./data/fl\_models                                | Directory to load the FL model from (relative to the project root)          |
| fl\_model\_load.load\_name=ResNet50\_no-avgpool\_out-100\_cifar100\_fine-tuned | FL model filename (without the .pth suffix)                                 |
| fl\_model\_load.load\_fc=False                                            | Do not load the classifier for the FL model (use initialized parameters)    |
| fl\_model\_kwargs.output\_length=1000                                     | Specify the number of classes for the FL model                              |
| fl\_model\_kwargs.is\_avgpool=False                                       | Remove the average pooling layer from the FL model (required)               |
| attacks.ICA-GA.model\_save\_dir=./data/gen\_models                        | Directory to load the generator model from                                  |
| attacks.ICA-GA.model\_save\_name=gen\_1x-CT-RCSE\_r50-fine-tuned           | Generator model filename (without the .pth suffix)                          |
| data\_iter\_kwargs.dataset\_name=cifar100                                 | Test dataset. Options: caltech256, cifar10, cifar100, mnist, imagenet      |
| data\_iter\_kwargs.batch\_size=8                                          | Test batch size                                                             |
| data\_iter\_kwargs.local\_steps=4                                         | Number of local training steps                                              |
| data\_iter\_kwargs.local\_epochs=1                                        | Number of local training epochs                                             |
| seed=42                                                                | Main random seed                                                            |
| data\_iter\_kwargs.dataloader\_seed=42                                    | Random seed for loading test data                                           |
| test\_batchs=10                                                         | Number of times to repeat the experiment                                    |
| device=cuda:0                                                          | Device for the experiment                                                   |

## 3. Generator Training

You can train or fine-tune the generator for different FL models.

### 3.1. Prepare the ImageNet Dataset

Generator training requires `ImageNet (ILSVRC2012)` as an auxiliary dataset. Please visit [image-net.org](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) to manually download the following files and extract them to `./data/datasets/imagenet`.

- `ILSVRC2012_devkit_t12.tar.gz`
- `ILSVRC2012_img_train.tar`
- `ILSVRC2012_img_val.tar`

The final directory structure should be as follows:

```
./data
└ datasets
    └ imagenet
        ├ ILSVRC2012_devkit_t12
        ├ train
        │  ├ n01440764
        │  └ ...
        └ val
            ├ n01440764
            └ ...
```

### 3.2. Run Training

#### Training the generator from scratch:

```sh
python3 exps/train_generator/train_generator.py \
    -result_dir=results\
    -fl_model_load.load_dir=./data/fl_models\
    -fl_model_load.load_name=ResNet50_no-avgpool_out-100_cifar100_fine-tuned\
    -generator_model_kwargs.save_dir=./data/gen_models\
    -generator_save_name=gen_1x-CT-RCSE_r50_epochs=end\
    -trainer_kwargs.train_dataset=imagenet\
    -trainer_kwargs.test_dataset=cifar100\
    -trainer_kwargs.batch_size=32\
    -trainer_kwargs.num_epochs=100\
    -trainer_kwargs.max_batchs=1000\
    -trainer_kwargs.lr=0.0001\
    -trainer_kwargs.loss.mse=1.0\
    -trainer_kwargs.loss.feat=0.01\
    -trainer_kwargs.loss.tv=0.01\
    -trainer_kwargs.loss.ssim=0.01\
    -trainer_kwargs.save_eval=psnr\
    -trainer_kwargs.save_eval_up=true\
    -trainer_kwargs.stop_eval_value=23\
    -trainer_kwargs.early_stop_patience=20
```

#### Incremental fine-tuning:

If you need to fine-tune an existing generator (e.g., after the FL model has been updated), add the `-generator_load_name` argument. It is recommended to set a smaller learning rate `trainer_kwargs.lr` and a finer evaluation granularity `trainer_kwargs.max_batchs`. This corresponds to the efficient persistent attack strategy proposed in our paper.

Example: (Assume `fl_epochs` represents the FL training round. You will need to prepare the corresponding FL model training history files beforehand.)

```sh
for fl_epochs in {1..50}; do
    python3 exps/train_generator/train_generator.py \
        -result_dir=results\
        -fl_model_load.load_dir=./data/fl_models\
        -fl_model_load.load_name=ResNet50_epochs=${fl_epochs}\
        -generator_model_kwargs.save_dir=./data/gen_models\
        -generator_load_name=gen_1x-CT-RCSE_r50_epochs=$((fl_epochs-1))\
        -generator_save_name=gen_1x-CT-RCSE_r50_epochs=${fl_epochs}\
        -trainer_kwargs.train_dataset=imagenet\
        -trainer_kwargs.test_dataset=cifar100\
        -trainer_kwargs.batch_size=32\
        -trainer_kwargs.num_epochs=100\
        -trainer_kwargs.max_batchs=100\
        -trainer_kwargs.lr=0.00001\
        -trainer_kwargs.loss.mse=1.0\
        -trainer_kwargs.loss.feat=0.01\
        -trainer_kwargs.loss.tv=0.01\
        -trainer_kwargs.loss.ssim=0.01\
        -trainer_kwargs.save_eval=psnr\
        -trainer_kwargs.save_eval_up=true\
        -trainer_kwargs.stop_eval_value=23
done
```

| Basic Argument                                                         | Description                                                                                                                                              |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| result\_dir=results                                                     | Directory to save training logs (relative to train\_generator.py)                                                                                         |
| recorder\_save\_name=result.csv                                          | [Optional] Training log filename. Defaults to `{generator_save_name}.csv`                                                                                |
| fl\_model\_load.load\_dir=./data/fl\_models                                | Directory to load the FL model from                                                                                                                      |
| fl\_model\_load.load\_name=ResNet50\_no-avgpool\_out-100\_cifar100\_fine-tuned | FL model filename (without the .pth suffix)                                                                                                              |
| generator\_model\_kwargs.save\_dir=./data/gen\_models                      | Directory for the generator model                                                                                                                        |
| generator\_load\_name=gen\_1x-CT-RCSE\_r50-fine-tuned                      | [Optional] Load an existing generator file for incremental fine-tuning                                                                                   |
| generator\_save\_name=gen\_1x-CT-RCSE\_r50\_1                               | Filename to save the generator model (without the .pth suffix)                                                                                           |
| trainer\_kwargs.train\_dataset=imagenet                                  | Training set for the generator                                                                                                                           |
| trainer\_kwargs.test\_dataset=cifar100                                   | Evaluation set for the generator                                                                                                                         |
| trainer\_kwargs.batch\_size=32                                           | Training batch size. Can be set to a maximum of 48 with 24GB of VRAM                                                                                     |
| trainer\_kwargs.num\_epochs=100                                          | Maximum number of training epochs                                                                                                                        |
| trainer\_kwargs.max\_batchs=1000                                         | [Optional] Maximum number of batches per epoch. Lowering this reduces training granularity for more frequent evaluations.                                |
| trainer\_kwargs.lr=0.0001                                               | Learning rate. `0.0001` for training from scratch, `0.00001` for incremental fine-tuning.                                                                    |
| trainer\_kwargs.loss.mse=1.0                                            | Loss weight                                                                                                                                              |
| trainer\_kwargs.loss.feat=0.01                                          | Loss weight                                                                                                                                              |
| trainer\_kwargs.loss.tv=0.01                                            | Loss weight                                                                                                                                              |
| trainer\_kwargs.loss.ssim=0.01                                          | Loss weight                                                                                                                                              |
| trainer\_kwargs.save\_eval=psnr                                          | Primary evaluation metric. Saves the model when the current metric is better than the best score so far.                                                 |
| trainer\_kwargs.save\_eval\_up=true                                       | Indicates that a higher evaluation metric is better.                                                                                                     |
| trainer\_kwargs.stop\_eval\_value=23                                      | [Optional] Early stopping value. Training stops when the metric reaches this value.                                                                      |
| trainer\_kwargs.early\_stop\_patience=10                                  | [Optional] Early stopping patience. Training stops if there is no improvement for this many consecutive epochs.                                        |

## 4. Citing Our Work

If you use our code or methods in your research, please consider citing our paper:

```bibtex
@article{li2025unmixing,
  title = {Unmixing gradients: Uncovering persistent leakage in Federated Learning via Independent Component Analysis},
  author={Li, Minhao and Wang, Le and Li, Zhaohua and Hu, Rongxin and Zhou, Tang and Fang, Binxing},
  journal = {Journal of Systems Architecture},
  pages = {103681},
  year = {2025},
  issn = {1383-7621},
  doi = {https://doi.org/10.1016/j.sysarc.2025.103681},
  url = {https://www.sciencedirect.com/science/article/pii/S1383762125003534}
}
```

## 5. Acknowledgements

Our work is built upon many excellent prior studies. We especially thank the following papers and open-source projects for providing valuable references and convenience:

> Fast Generation-Based Gradient Leakage Attacks: An Approach to Generate Training Data Directly From the Gradient
- https://ieeexplore.ieee.org/abstract/document/10505158
- https://github.com/pigeon-dove/FGLA

> Cocktail Party Attack: Breaking Aggregation-Based Privacy in Federated Learning using Independent Component Analysis
- https://proceedings.mlr.press/v202/kariyappa23a.html
- https://github.com/facebookresearch/cocktail_party_attack

## 6. License

This project is licensed under the [MIT License](https://github.com/liminhao-99/ICA-GA/blob/main/LICENSE).