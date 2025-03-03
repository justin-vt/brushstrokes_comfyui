from setuptools import setup, find_packages

setup(
    name='ComfyUI-brushstrokes',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['Pillow', 'wand', 'gmic-py'],
    description='A ComfyUI node for painterly brush strokes using ImageMagick and G\'MIC',
    author='Justin Pinder',
    url='https://github.com/justin-vt/comfyui-brushstrokes',
)
