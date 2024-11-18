# Preprocessing Pathology Reports for Vision-Language Model Development
This repository contains all code to support the paper:  

***"Preprocessing Pathology Reports for Vision-Language Model Development"***.

Accepted at the MICCAI 2024 COMPAYL workshop.

[[`PLMR`](https://proceedings.mlr.press/v254/lucassen24a.html)] [[`OpenReview`](https://openreview.net/forum?id=SUgnMdiJ2q)] [[`Poster`](https://github.com/RTLucassen/report_preprocessing/tree/main/.github/poster.pdf)]

<div align="center">
  <img width="100%" alt="Overview" src=".github\overview.png">
</div>

## Model Parameters
The parameters of the **Dutch to English translation model** ([`Repo`](https://huggingface.co/RTLucassen/opus-mt-nl-en-finetuned-melanocytic-lesion-reports)) and **subsentence segmentation model** ([`Repo`](https://huggingface.co/RTLucassen/flan-t5-large-finetuned-melanocytic-lesion-reports)) are available from the corresponding HuggingFace repositories. 

## Citing
If you found our work useful in your research, please consider citing our paper:
```
@inproceedings{lucassen2024preprocessing,
  title={Preprocessing Pathology Reports for Vision-Language Model Development},
  author={Lucassen, Ruben T. and Luijtgaarden, Tijn van de and Moonemans, Sander P. J. and Blokx, Willeke A. M. and Veta, Mitko},
  booktitle={Proceedings of the MICCAI Workshop on Computational Pathology},
  pages={61--71},
  year={2024},
  editor={Ciompi, Francesco and Khalili, Nadieh and Studer, Linda and Poceviciute, Milda and Khan, Amjad and Veta, Mitko and Jiao, Yiping and Haj-Hosseini, Neda and Chen, Hao and Raza, Shan and Minhas, FayyazZlobec, Inti and Burlutskiy, Nikolay and Vilaplana, Veronica and Brattoli, Biagio and Muller, Henning and Atzori, Manfredo and Raza, Shan and Minhas, Fayyaz},
  volume={254},
  series={Proceedings of Machine Learning Research},
  month={06 Oct},
  publisher={PMLR},
}
```




