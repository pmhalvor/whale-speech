import setuptools

setuptools.setup(
    name="whale-speech",
    version="0.0.1",
    packages=setuptools.find_packages(),
    intall_requires=[
        "tensorflow",
        "tensorflow_hub",
        "numpy",
        "scipy",
        "matplotlib",
        "apache_beam"
        "git+https://github.com/open-oceans/happywhale.git",
        "pytest",
        "pyyaml",
        "six",
        "librosa",
        "soundfile",
        "apache-beam[gcp]"
    ]
)