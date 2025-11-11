SPIKAN r = 20
```sh
!python taylor_couette2d.py --model spikan --features 20 --r 20 --lr 2e-4 --epochs 50000 --lbda_b 100 --lr_decay_steps 2500 --lr_decay_rate 0.9
```

SPIKAN r = 3
```sh
!python taylor_couette2d.py --model spikan --features 20 --r 3 --lr 2e-4 --epochs 50000 --lbda_b 100 --lr_decay_steps 2500 --lr_decay_rate 0.9
```

SPINN r = 20
```sh
!python taylor_couette2d.py --model spinn --features 20 --r 20 --lr 2e-4 --epochs 50000 --lbda_b 100 --lr_decay_steps 2500 --lr_decay_rate 0.9
```

SPINN r = 3
```sh
!python taylor_couette2d.py --model spinn --features   20 --r 3 --lr 2e-4 --epochs 50000 --lbda_b 100 --lr_decay_steps 2500 --lr_decay_rate 0.9
```
SPINN Cartesian r = 20
```sh
!python taylor_couette2d_cartesian.py --model spinn --features 20 --r 20 --lr 2e-4 --epochs 50000 --lbda_b 100
```