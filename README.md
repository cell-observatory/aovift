<div align="center">
  <img class="center" src="https://www.dropbox.com/scl/fi/xonddbzyjptsh1c3me0y5/fish.gif?rlkey=7owu8ez9iuk1dyabbkj4idhrz&raw=1" />
</div>

AOViFT: Adaptive Optical Vision Fourier Transformer 
====================================================
***Fourier-Based 3D Multistage Transformer for Aberration Correction in Multicellular Specimens***

[![python](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=3776AB)](https://www.python.org/)
[![tensorflow](https://img.shields.io/badge/tensorFlow-2.14+-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![license](https://img.shields.io/github/license/cell-observatory/aovift.svg?style=flat&logo=git&logoColor=white)](https://opensource.org/license/bsd-2-clause/)
[![issues](https://img.shields.io/github/issues/cell-observatory/aovift.svg?style=flat&logo=github)](https://github.com/cell-observatory/aovift/issues)
[![pr](https://img.shields.io/github/issues-pr/cell-observatory/aovift.svg?style=flat&logo=github)](https://github.com/cell-observatory/aovift/pulls)

<div align="center">
  <img class="center" src="https://www.dropbox.com/scl/fi/zc2b1qqd7wte2rxzw3qtg/model.png?rlkey=n7gtkbs6rq8jjk3mr9gwxc5zz&raw=1" />
</div>

# Table of Contents

* [Benchmark](#benchmark)
* [Examples](#examples)
* [Installation](#installation)
* [Pretrained models](#pretrained-models)
* [BibTeX](#bibtex)
* [License](#license)


## Benchmark
<div align="center">
  <img src="https://www.dropbox.com/scl/fi/vq099qein6juyekng8w5n/eval.png?rlkey=8hy7flhb78n1evv5w38xkdh1r&raw=1" />
  <img class="center" src="https://www.dropbox.com/scl/fi/5psg2uunus1xesa8doz28/benchmark.png?rlkey=iq2gbmnpn6idm1pc2k5fmmq6x&raw=1" />
</div>



## Examples
The [`src/python ao.py`](src/python ao.py) script provides a CLI
for running our models on a given 3D stack (`.tif` file).


<div align="center">
  <img src="https://www.dropbox.com/scl/fi/e1f0kpnoofa10moi85zvv/ap2.gif?rlkey=3pvphchl69dxgk5k72njt8brc&raw=1" />
  <img src="https://www.dropbox.com/scl/fi/d8izku9dds87b18ctftfv/mitochondria.gif?rlkey=x13cc4lolp3bcycuhzjqovlrv&raw=1" />
</div>


<div align="center">
  <img src="https://www.dropbox.com/scl/fi/dj4mgimnzljxih73q07zh/fishmap.png?rlkey=ijzjsttea5xa7a9m4ptypvy8c&raw=1" />
</div>


## Installation

### Git Clone repository to your host system
```shell
# Please make sure you have Git LFS installed to download our pretrained models
git clone --recurse-submodules https://github.com/cell-observatory/aovift.git
# ...to later update to the latest, greatest
git pull --recurse-submodules
```

### Docker [image]()
Our prebuilt image with Python, TensorFlow, and all packages installed for you.
```shell
# 
docker pull 
```
If you want to run a local version of the image, see the [Dockerfile](https://github.com/cell-observatory/aovift/blob/main/Dockerfile)

## Pretrained [models](https://www.dropbox.com/scl/fo/yqr5nnmyfjoz53e4cav4d/AE4EDwrkOIytICIy7yDK6J4?rlkey=hm3em3yow48p390n8jvrt0jly&st=whj5il8d&dl=0)

All pre-trained models can be downloaded from our [pretrained models repository](https://www.dropbox.com/scl/fo/yqr5nnmyfjoz53e4cav4d/AE4EDwrkOIytICIy7yDK6J4?rlkey=hm3em3yow48p390n8jvrt0jly&st=whj5il8d&dl=0).

If you wish to download all of our models at once, 
you can use this [link](https://www.dropbox.com/scl/fo/yqr5nnmyfjoz53e4cav4d/AE4EDwrkOIytICIy7yDK6J4?rlkey=hm3em3yow48p390n8jvrt0jly&st=whj5il8d&raw=1) and extract the desired *.h5 file from the zip file.

## BibTeX

```bibtex
comming soon
```

## License 

This work is licensed under the [BSD 2-Clause License](https://github.com/cell-observatory/aovift/blob/main/LICENSE)