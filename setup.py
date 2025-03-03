from setuptools import setup, find_packages

setup(
    name='brush-strokes-comfyui',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['Pillow', 'wand', 'gmic-py'],
    description='A ComfyUI node for painterly brush strokes using ImageMagick and G\'MIC',
    author='Your Name',
    url='https://github.com/YourUser/brush-strokes-comfyui',
)
