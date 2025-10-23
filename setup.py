from setuptools import setup , find_packages
from typing import List 

# read README.md file
with open("README.md" , 'r' , encoding = "utf-8") as f:
    long_description = f.read()


HYPER_E_DOT = "-e ."


# read the requirements.txt file
def get_requirements(file_path: str = "requirements.txt") -> List[str]:
    requirements = []
    
    with open(file_path , 'r') as file:
        requirements = file.readlines()
        
        # remove the \n
        requirements = [req.replace("\n" , "") for req in requirements]
        
        # remove -e . if exists
        if HYPER_E_DOT in requirements:
            requirements.remove(HYPER_E_DOT)
    
    return requirements


short_description = (
    "This project is an end-to-end ML system that predicts laptop prices from specs like processor, RAM, storage, GPU, and brand," +
    " covering data collection, preprocessing, model training, evaluation, and deployment with practical feature engineering and pipeline design."
)


# All the meta data
project_name = "laptopPrice"
version = "0.0.1"
author_name = "Tipto_Ghosh"
author_email = "tiptoghosh@gmail.com"
project_url = "https://github.com/Tipto-Ghosh/Predict-Laptops-Price"


setup(
    name = project_name,
    version = version,
    description = short_description,
    long_description = long_description,
    long_description_content_type = "text/markdown",
    maintainer = author_name,
    maintainer_email = author_email,
    author = author_name,
    author_email = author_email,
    url = project_url,
    packages = find_packages(),
    install_requires = get_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)