from setuptools import setup,find_packages

setup(
    name = "AI-OZER",
    version = "0.0.1",
    packages = find_packages(),
    install_requires = [
        "torch",
        "ultralytics",
        "cv2"
    ],
    entry_points = {
        "console_sripts" : ["ahmet = AI-OZER:ahmet"]
    }
)