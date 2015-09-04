Deep Learning + Arts = ArtsyNetworks
=======

*The system uses neural representations to separate
and recombine content and style of arbitrary images, providing a neural
algorithm for the creation of artistic images.* - Gatys et al.

This is an humble attempt to implement the algorithm described in [http://arxiv.org/abs/1508.06576](http://arxiv.org/abs/1508.06576>) by Gatys, Ecker and Bethge (first submitted on 26 August 2015). The code is inspired by *Lasagne Recipe* - [styletransfer](https://github.com/Lasagne/Recipes/blob/master/examples/styletransfer/Art%20Style%20Transfer.ipynb), yet has several modifications.
The pretrained network is downloaded from [https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl) .

Dependencies:

* theano=0.7.0
* lasagne=0.2.dev1
* skimage=0.11.3
* matplotlib=1.4.3

NVIDIA cuDNN is also required: [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

Code is tested to work with Python 2.7 under Ubuntu 14.04 and Windows 8.1 v6.3.9600 (both 64 bit) with GeForce GT 755M. 


How to Run?
------------

From command line:

```
  python art_it_up.py base_image_path style_image_path
```

E.g.

```
  python art_it_up.py images/tietotalo.jpg images/the_starry_night.jpg
```

Examples
--------

Base image (Tietotalo - TTY):
<br>
<a href="url"><img src="https://raw.githubusercontent.com/ogencoglu/ArtsyNetworks/master/images/tietotalo.JPG" align="left" width="240" ></a>
<br>


Style image (The Starry Night - Van Gogh):
<br>
<a href="url"><img src="https://raw.githubusercontent.com/ogencoglu/ArtsyNetworks/master/images/the_starry_night.jpg" align="left"  width="240" ></a>


<br>
Result:
<a href="url"><img src="https://raw.githubusercontent.com/ogencoglu/ArtsyNetworks/master/images/neural_painting.png" align="left"  width="240" ></a>

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

Base image (Me):
<br>
<a href="url"><img src="https://raw.githubusercontent.com/ogencoglu/ArtsyNetworks/master/images/sakalli_small.jpg" align="left" width="240" ></a>
<br>


Style image (The Scream - Munch):
<br>
<a href="url"><img src="https://raw.githubusercontent.com/ogencoglu/ArtsyNetworks/master/images/scream.jpg" align="left"  width="240" ></a>


<br>
<br>
<br>
<br>
<br>
<br>
Result:
<a href="url"><img src="https://raw.githubusercontent.com/ogencoglu/ArtsyNetworks/master/images/neural_painting_ouz.png" align="left"  width="240" ></a>
