from setuptools import setup,find_packages

setup(name= "pixellib",
      version='0.2.1',
      description='PixelLib is a library for performing semantic and instance segmentation of images and videos using few lines of code.',
      url="https://github.com/ayoolaolafenwa/PixelLib",
      author='Ayoola Olafenwa',
      license='MIT',
      packages= find_packages(),
      install_requires=['pillow','scikit-image','opencv-python'],
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
      )
