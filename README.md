<div align="center">

$\Large \boldsymbol{\mathsf{\color{#6366f1}M\color{#a855f7}3\color{#ec4899}\text{-}TTS}}: \mathsf{\color{#de2910}M\color{black}\text{ulti-}\color{#de2910}M\color{black}\text{odal\ DiT\ Alignment}\ \color{black}\\&\ \color{#de2910}M\color{black}\text{el-latent}}$

</div>


<div align="center">

[![Findings of EMNLP 2026](https://img.shields.io/badge/Findings%20of%20EMNLP-2026-blue)](https://aclanthology.org/2026.findings-emnlp.240.pdf)
[![Demo Page](https://img.shields.io/badge/Demo-Page-blue)](https://wwwwxp.github.io/M3-TTS-Demo/)
<!-- [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](Demo_Link_Here) -->

</div>

## 📅 Roadmap

- [x] Release model code
- [ ] Release training and inference code
- [ ] Release pre-trained model weights

## 🔥 Key Features

- **No Pseudo-Alignment:** Achieves stable alignment implicitly via Joint-DiT attention.
- **Mel-VAE Codec:** Efficient latent representation for faster training and high-fidelity reconstruction.
- **Unified Architecture:** A simple, end-to-end framework without complex multi-stage pipelines.

## 🙌 Acknowledgements

This project is built upon the excellent work of **[F5-TTS](https://github.com/SWivid/F5-TTS)**, **[MMAudio](https://github.com/hkchengrex/MMAudio)** and **[Zip-Voice](https://github.com/k2-fsa/ZipVoice)**. We thank the authors for their open-source contributions.

## 📝 Citation

If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝:

```bibtex
@inproceedings{wang2026m3,
  title={M3-TTS: Native Two-Stream Multi-Modal Diffusion Transformer for Zero-Shot Speech Synthesis},
  author={Wang, Xiaopeng and Qiang, Chunyu and Fu, Ruibo and Wen, Zhengqi and Guo, Yuxin and Liu, Xuefei and Liu, Yukun and Liang, Yuzhe and Yin, Kang and Ma, Teng and others},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  pages={5061--5071},
  year={2026}
}
