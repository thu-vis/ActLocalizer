# ActLocalizer

This is the source code for our paper "Enhancing Single-Frame Supervision for Better Temporal Action Localization."

## Install environment and run

### Option 1: Docker (recommended)
1. The easiest way to install a environment to run the demo is to use docker. The image `chencjgene/actlocalizer-run:latest` contains the source codes and data, and have the dependencies installed. You can pull and run the image by:

```sh
$ docker pull chencjgene/actlocalizer-run:latest
```
2. Run the docker image:
   
```sh
$ docker run -p 20222:20222 -p 30221:30221 -it chencjgene/actlocalizer-run:latest
```

3. Run backend

```sh
$ cd /root/ActLocalizer/
$ nohup python manager.py run 30221 &
```

4. Run frontend

```sh
$ cd vis
$ npm install (it will take a while)
$ npm run serve
```

5. Visit http://localhost:20222/ in a browser.



### Option 2: Install with python and node.js
1. This project uses [python 3.8](https://www.python.org/). Go check it out if you don't have it installed.

2. install python package.
```sh
$ pip install -r requirements.txt
$ pip install torch
```
3. Download the repo

4. Download demo data from [here](https://drive.google.com/file/d/1EZ6ivfi4xJVphaY0WPkMsOm3SGuD-uDR/view?usp=sharing), and unpack it in the root folder 

5. Run backend

```sh
$ nohup python manager.py run 30221 &
```

6. Run frontend

```sh
$ cd vis
$ npm install (it will take a while)
$ npm run serve
```

7. Visit http://localhost:20222/ in a browser.



## Contact
If you have any problem about this code, feel free to contact
- changjianchen.me@gmail.com

or describe your problem in Issues.
