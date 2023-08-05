# ActLocalizer

This is the source code for our paper "Enhancing Single-Frame Supervision for Better Temporal Action Localization."

## Install environment

### Docker
The easiest way to install a environment to run the demo is to use docker. The image `chencjgene/actlocalizer-run:latest` contains the source codes and data, and have the dependencies installed. You can pull and run the image by:

```sh
$ docker pull chencjgene/actlocalizer-run:latest
$ docker run -it chencjgene/actlocalizer-run:latest
```

### Install with python and node.js
1. This project uses [python 3.8](https://www.python.org/). Go check it out if you don't have it installed.

2. install python package.
```sh
$ pip install -r requirements.txt
$ pip install torch
```

3. install nodejs package: check `README.md` under `vis` for more details.

## Run
Make sure the ports 30221 and 20222 are not used.

1. run backend. If you want to change the port `30221`, please change the setting in `vis/src/store/index.js` accordingly.
```sh
python manager.py run 30221
```

2. run frontend: check `README.md` under `vis` for more details.

## Contact
If you have any problem about this code, feel free to contact
- changjianchen.me@gmail.com

or describe your problem in Issues.
