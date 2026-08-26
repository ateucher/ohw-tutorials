# Underwater Machine Vision: Object Detection and Segmentation

This tutorial uses FathomNet imagery to introduce object detection, domain shift, YOLO fine-tuning, class-specific detection, and image segmentation. The main guided path focuses on bounding boxes; segmentation and class-specific extensions remain available in the notebook and appendices.

[Open the notebook](underwater_machine_vision_object_detection_and_segmentation.ipynb) or [open it in Google Colab](https://colab.research.google.com/github/oceanhackweek/ohw-tutorials/blob/OHW26/02-Wed/UnderwaterMachineVision/underwater_machine_vision_object_detection_and_segmentation.ipynb).

## Runtime notes

A GPU is helpful but not required: CPU-only participants use saved results for the training sections by default, while the inference exercises run live. The notebook also includes an opt-in switch for trying the small YOLO fine-tune on a CPU.

The local data archive is unpacked automatically on first use. Internet access is still needed to download pretrained model checkpoints and to use the optional live SAM3 path.

## Included files

- `data/fathomnet_underwater_tutorial_bundle.zip`: compact images and annotations used by the exercises
- `data/reference_training/`: saved training metrics and segmentation examples for CPU-only runtimes
- `scripts/`: data preparation, runtime, visualization, and SAM3 helpers imported by the notebook
- `images/`: figures displayed in the tutorial instructions
- `requirements.txt`: pinned core machine-learning packages and supporting libraries

The imagery is derived from the [FathomNet Database](https://database.fathomnet.org/fathomnet/#/). Review the [FathomNet data-use policy](https://www.fathomnet.org/datause) and the bundle's `licenses/attribution.csv` before redistributing it.
