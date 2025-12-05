<div align="center">

$\Large \boldsymbol{\mathsf{\color{#6366f1}M\color{#a855f7}3\color{#ec4899}\text{-}TTS}}: \mathsf{\color{#de2910}M\color{black}\text{ulti-}\color{#de2910}M\color{black}\text{odal\ DiT\ Alignment}\ \color{black}\\&\ \color{#de2910}M\color{black}\text{el-latent}}$

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2512.04720)
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
@article{wang2025m3tts,
  title={M3-TTS: Multi-modal DiT Alignment \& Mel-latent for Zero-shot High-fidelity Speech Synthesis},
  author={Wang, Xiaopeng and Qiang, Chunyu and Fu, Ruibo and others},
  journal={arXiv preprint arXiv:2512.04720},
  year={2025}
}
