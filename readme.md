BrainEncoder 

A self-supervised tool for brain region definition and segmentation.

Setup

Clone the repository, change into the 'BrainEncoder' folder.

If you are on the Chunglab Computational server you can copy the relevant test image via
    cp /media/ssdshare1/general/computational_projects/brain_segmentation/NeuNBrainSegment_compressed/NeuNBrainSegment_compressed.tiff .

Alternatively you can download it from

    https://1drv.ms/i/s!AiJklHUkocm3y_ZZatAFjUrPKQ8vWQ?e=E20RVk

and copy it into the BrainEncoder folder.


Ensure that '- cudatoolkit=<version>' in 'environment.yml' matches the installed version of CUDA on your system.

Setup the conda environment via

    conda env create -f environment.yml -y
    conda activate brainEncoder_env

Train the autoencoder via

    python trainBrainEncoder.py

Compare input and output images via

    python showBrainEncoderPerformance.py

Visualize latent space clustering via

    cd BrainEncoder; python latentSpaceVisualization.py
