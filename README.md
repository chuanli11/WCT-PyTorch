# WCT-PyTorch
Make PyTorch WCT working again


### Prerequisite

- CUDA: 10.0
- CUDNN: 7.6.5

Install PyTorch 1.4.0 in virtualenv
```
wget https://download.pytorch.org/whl/cu100/torch-1.4.0%2Bcu100-cp36-cp36m-linux_x86_64.whl
wget https://files.pythonhosted.org/packages/7e/90/6141bf41f5655c78e24f40f710fdd4f8a8aff6c8b7c6f0328240f649bdbe/torchvision-0.5.0-cp36-cp36m-manylinux1_x86_64.whl
virtualenv -p /usr/bin/python3.6 venv && . venv/bin/activate && find . -maxdepth 1 -name "*.whl" | xargs pip install && rm *.whl
```

### Usage

```
python main.py --content=images/content/in1.jpg --style=images/style/in1.jpg --output=output/1.jpg --alpha=1.0  --gpu=0 --content_w=512 --content_h=512 --style_w=512 --style_h=512

python main.py --content=images/content/in1.jpg --style=images/style/in1.jpg --output=output/2.jpg --alpha=.5  --gpu=0 --content_w=512 --content_h=512 --style_w=512 --style_h=512

python main.py --content=images/content/in3.jpg --style=images/style/in4.jpg --output=output/3.jpg --alpha=1.0  --gpu=0 --content_w=1024 --content_h=1024 --style_w=1024 --style_h=1024
```

Results saved in `output`