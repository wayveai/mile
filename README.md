# MILE
This is the PyTorch implementation for inference and training of the world model and driving policy as 
described in:

> **Model-Based Imitation Learning for Urban Driving**
>
> [Anthony Hu](https://anthonyhu.github.io/), 
[Gianluca Corrado](https://github.com/gianlucacorrado),
[Nicolas Griffiths](https://github.com/nicolasgriffiths), 
[Zak Murez](http://zak.murez.com/),
[Corina Gurau](https://github.com/cgurau),
[Hudson Yeo](https://github.com/huddyyeo), 
[Alex Kendall](https://alexgkendall.com/),
[Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/index.htm) and
[Jamie Shotton](https://jamie.shotton.org/). 
>
> [NeurIPS 2022](https://arxiv.org/abs/2210.07729)<br/>
> [Blog post](https://wayve.ai/blog/model-based-imitation-learning-for-urban-driving)

<p align="center">
     <img src="https://github.com/wayveai/mile/releases/download/v1.0/mile_driving_in_imagination.gif" alt="MILE driving in imagination">
     <br/> Our model can drive in the simulator with a driving plan predicted entirely from imagination.
     <br/> From left to right we visualise: RGB input, ground truth bird's-eye view semantic segmentation,
     predicted bird's-eye view segmentation.
     <br/> When the RGB input becomes sepia-coloured, the model is driving in imagination.
     <sub><em>
    </em></sub>
</p>

If you find our work useful, please consider citing:
```bibtex
@inproceedings{mile2022,
  title     = {Model-Based Imitation Learning for Urban Driving},
  author    = {Anthony Hu and Gianluca Corrado and Nicolas Griffiths and Zak Murez and Corina Gurau
   and Hudson Yeo and Alex Kendall and Roberto Cipolla and Jamie Shotton},
  booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
  year = {2022}
}
```

## ⚙ Setup
- Create the [conda](https://docs.conda.io/en/latest/miniconda.html) environment by running `conda env create`.

## 🏄 Evaluation

- Download [CARLA 0.9.11](https://github.com/carla-simulator/carla/releases/tag/0.9.11).
- Download the model [pre-trained weights](https://github.com/wayveai/mile/releases/download/v1.0/mile.ckpt).
- Run `bash run/evaluate.sh ${CARLA_PATH} ${CHECKPOINT_PATH} ${PORT}`, with 
 `${CARLA_PATH}` the path to the CARLA .sh executable,
`${CHECKPOINT_PATH}` the path to the 
pre-trained weights, and `${PORT}` the port to run CARLA (usually `2000`).

## 📖 Data Collection
- Run `bash run/data_collect.sh ${CARLA_PATH} ${DATASET_ROOT} ${PORT}`, with 
 `${CARLA_PATH}` the path to the CARLA .sh executable,
`${DATASET_ROOT}` the path where to save data, and `${PORT}` the port to run CARLA (usually `2000`).

## 🏊 Training
To train the model from scratch:
- Organise the dataset folder as described in [DATASET.md](DATASET.md).
- Activate the environment with `conda activate mile`.
- Run `python train.py --config mile/configs/mile.yml DATASET.DATAROOT ${DATAROOT}`, with `${DATAROOT}`
the path to the dataset.

## 🙌 Credits
Thanks to the authors of [End-to-End Urban Driving by Imitating a Reinforcement Learning Coach](https://github.com/zhejz/carla-roach)
for providing a gym wrapper around CARLA making it easy to use, as well as an RL expert to collect data.
