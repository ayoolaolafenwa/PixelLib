from setuptools import setup,find_packages

setup(name= "pixellib",
      version='0.1.0',
      description='Pixellib is a library for performing semantic and instance segmentation of images using few lines of code.',
      url="https://github.com/ayoolaolafenwa/PixelLib",
      author='Ayoola Olafenwa',
      license='MIT',
      packages= find_packages(),
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
      )
