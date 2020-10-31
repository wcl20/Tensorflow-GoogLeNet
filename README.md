# Tensorflow MiniGoogLeNet

Trained a variant of GoogLeNet model on Tiny ImageNet dataset.

## Setup
Generate Virtual environment
```bash
python3 -m venv ./venv
```
Enter environment
```bash
source venv/bin/activate
```
Install required libraries
```bash
pip install -r requirements.txt
```
Install Dataset [Tiny Imagenet](https://tiny-imagenet.herokuapp.com/)

Setup config file to dataset.

Build project
```bash
python3 build.py
```
Train model
```bash
python3 train.py --lr 1e-3 --epochs 50
python3 train.py --model <path to model> --start 40 --lr 1e-4 --epochs 30
python3 train.py --model <path to model> --start 60 --lr 1e-5 --epochs 10
```
Setup config file to model

Test model
```bash
python3 test.py
```
