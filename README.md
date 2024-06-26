# EOT-cpp

## Build from source

```Shell
git clone https://github.com/EricAlex/EOT-cpp.git
mkdir build && cd build
cmake ../EOT-cpp
make
```

## Simulation results

### One simple rectangular object

#### Static

![static rectangular object](img/static.gif)

Partially visible: 

![static rectangular object (partially visible)](img/part_static.gif)

#### Constant turning rate

![constant turning rate](img/constant_turning_rate.gif)

Partially visible: 

![constant turning rate (partially visible)](img/part_crot.gif)

#### Accelerated turning rate

![accelerated turning rate](img/accelerated_turning_rate.gif)

Partially visible: 

![accelerated turning rate (partially visible)](img/part_arot.gif)

### Multiple rectangular objects moving with random acceleration

- The dimensions of the objects are random;

- The number of measurements and the measurement position on the objects are random;

- The measurements are noisy with clutters.

![multiple rectangular objects moving with random acceleration](img/simulation.gif)

# References

1. Meyer, F., & Williams, J. L. (2021). Scalable detection and tracking of geometric extended objects. IEEE Transactions on Signal Processing, 69, 6283-6298.

1. Meyer, F., & Win, M. Z. (2020). Scalable data association for extended object tracking. IEEE Transactions on Signal and Information Processing over Networks, 6, 491-507.
