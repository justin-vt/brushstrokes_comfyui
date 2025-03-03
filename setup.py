from setuptools import setup, find_packages

setup(
    name='ComfyUI-brushstrokes',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Pillow>=9.0.0',
        'wand>=0.6.10',
        'gmic-py>=2.9.0',
    ],
    description='A ComfyUI node for painterly brush strokes using ImageMagick and G\'MIC',
    author='Justin Pinder',
    url='https://github.com/justin-vt/ComfyUI-brushstrokes',
    keywords=['ComfyUI', 'Image Processing', 'ImageMagick', 'Gmic'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.8'
)
