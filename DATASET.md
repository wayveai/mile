## 📖 Dataset

- `${DATAROOT}` is a folder organised as follows. 
```
${DATAROOT}  
│
└───trainval
│    │
│    └───train
│    │     Town01
│    │     Town03
│    │     Town04
│    │     Town06
│    └───val
│          Town02
│          Town05
│     
└───mini
│    │
│    └───train
│    │     Town01
│    │     Town03
│    │     Town04
│    │     Town06
│    └───val
│          Town02
│          Town05
```

The content of in `Town0x` is collected with `run/data_collect.sh`. As an example:

```
Town01
│
└───0000
│    │
│    └───birdview
│    │     birdview_000000000.png
│    │     birdview_000000001.png
│    │     ..
│    └───image
│    │     image_000000000.png
│    │     image_000000001.png
│    │     ..
│    └───routemap
│    │     routemap_000000000.png
│    │     routemap_000000001.png
│    │     ..
│    └───pd_dataframe.pkl
│     
└───0001
``` 

Each folder `0000`, `0001` etc. contains a run collected by the [RL expert](https://github.com/zhejz/carla-roach).