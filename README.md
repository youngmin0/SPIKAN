<a id="readme-top"></a>
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/youngmin0/SPIKAN">
  </a>


  <p align="center">
    SPIKAN, SPINN의 랭크를 자동 선택해주는 프레임 워크입니다.
    <br />
    <a href="https://github.com/youngmin0/SPIKAN"><strong>GitHub »</strong></a>
    <br />
    <br />
  </p>
</div>



<br />

SPIKAN r = 20
```sh
!python taylor_couette2d.py --model spikan --features 20 --r 20 --lr 2e-4 --epochs 50000 --lbda_b 100 --lr_decay_steps 2500 --lr_decay_rate 0.9
```

SPIKAN r = 2
```sh
!python taylor_couette2d.py --model spikan --features 20 --r 2 --lr 2e-4 --epochs 50000 --lbda_b 100 --lr_decay_steps 2500 --lr_decay_rate 0.9
```

SPINN r = 20
```sh
!python taylor_couette2d.py --model spinn --features 20 --r 20 --lr 2e-4 --epochs 50000 --lbda_b 100 --lr_decay_steps 2500 --lr_decay_rate 0.9
```

SPINN r = 2
```sh
!python taylor_couette2d.py --model spinn --features 20 --r 2 --lr 2e-4 --epochs 50000 --lbda_b 100 --lr_decay_steps 2500 --lr_decay_rate 0.9
```
SPINN Cartesian r = 20
```sh
!python taylor_couette2d_cartesian.py --model spinn --features 20 --r 20 --lr 2e-4 --epochs 50000 --lbda_b 100
```
