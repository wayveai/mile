## π Dataset

- `${DATAROOT}` is a folder organised as follows. 
```
${DATAROOT}  
β
ββββtrainval
β    β
β    ββββtrain
β    β     Town01
β    β     Town03
β    β     Town04
β    β     Town06
β    ββββval
β          Town02
β          Town05
β     
ββββmini
β    β
β    ββββtrain
β    β     Town01
β    β     Town03
β    β     Town04
β    β     Town06
β    ββββval
β          Town02
β          Town05
```

The content of in `Town0x` is collected with `run/data_collect.sh`. As an example:

```
Town01
β
ββββ0000
β    β
β    ββββbirdview
β    β     birdview_000000000.png
β    β     birdview_000000001.png
β    β     ..
β    ββββimage
β    β     image_000000000.png
β    β     image_000000001.png
β    β     ..
β    ββββroutemap
β    β     routemap_000000000.png
β    β     routemap_000000001.png
β    β     ..
β    ββββpd_dataframe.pkl
β     
ββββ0001
``` 

Each folder `0000`, `0001` etc. contains a run collected by the [RL expert](https://github.com/zhejz/carla-roach).