# ML_project.github.io
The model in the DCGAN_CIFAR10.py (Our main code) is the model set up by the group project, 


and after modifying the path settings in the file, the GPU version of the pytorch framework can be run. 
The DCGAN_CIFAR-10.py file and DCGAN_CIFAR10 are used to test the ability of DCGAN to generate images, 
and the ability of the model to generate images is shown in the experimental section of the report, 
in addition, DCGAN_MNIST and GAN_MNIST are used to compare the gap between the ability of GAN and DCGAN models in generating images





Finally, the two datasets are : CIFAR10, MNIST. These two datasets can be downloaded directly from the dataset function in pytorch.
dataset = CIFAR10(root=path , download=True,transform=...)
