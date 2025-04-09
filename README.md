# Classifier-guided Gradient Modulation for Enhanced Multimodal Learning

[ACM MM 2025 Under Review] Official PyTorch implementation of the paper "Mitigating Modal Imbalance in Multimodal Emotion Recognition via Dynamic Gradient Shaping"

## Abstract

Multimodal emotion recognition in conversations aims to enhance model performance by integrating information from multiple modalities. However, current research primarily focuses on multimodal fusion and representation learning strategies, which overlook the under-optimized modality representations caused by imbalances in unimodal performance during joint learning. To address this issue, we first analyze the imbalance phenomenon at both the feed-forward and back-propagation stages with neural networks. Our experiments reveal that this imbalance arises from previously overlooked gradient conflicts between multimodal and unimodal learning objectives.
To well mitigate these conflicts, we propose a **D**ynamic **G**radient **S**haping (DGS) strategy. Specifically, we introduce a Shapley-based modality valuation metric to evaluate the contribution of each modality. Based on this assessment, our approach dynamically adjusts the gradient magnitude for each modality during training, ensuring balanced multimodal cooperation without suppressing unimodal learning. Additionally, we employ MMPareto*  algorithm, which ensures a final gradient direction aligns with all learning objectives, providing innocent unimodal assistance. Experimental results show that DGS improves the performance of all baseline models and outperforms existing plug-in methods.


<img width="4551" alt="framework" src="https://github.com/user-attachments/assets/1f966f0e-a597-43ec-a461-6ddb0e1521fb" />


## Getting Started

### Pre-trained Model

For feature extraction of Food 101 dataset, we use pre-trained BERT ([google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)) and ViT model ([google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)).  The pre-trained models are used only in Food 101 dataset.

### Dataset

The raw data can be found at [IEMOCAP](https://sail.usc.edu/iemocap/ "IEMOCAP") and [MELD](https://github.com/SenticNet/MELD "MELD").

In our paper, we use pre-extracted features. The multimodal features (including RoBERTa-based and GloVe-based textual features) are available at [here](https://www.dropbox.com/sh/4b21lympehwdg4l/AADXMURD5uCECN_pvvJpCAy9a?dl=0 "here").

We also provide some pre-trained checkpoints on RoBERTa-based IEMOCAP at [here](https://www.dropbox.com/sh/gd32s36v7l3c3u9/AACOipUURd7gEbEcdYSrmP-0a?dl=0 "here").

### Running the Code

Run the python file under the directory `run` according to the dataset.

For example, if you want to evaluate the performance on MELD dataset,  you can run the `train_OGM_Shapley_MMPareto_MELD.py` file:

```bash
nohup python -u train_OGM_Shapley_MMPareto_MELD.py --base-model 'GRU' --dropout 0.4 --lr 0.00001 --batch-size 8 --graph_type='hyper' --use_topic --epochs=50 --graph_construct='direct' --multi_modal --use_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' --norm BN --num_L=1 --num_K=1 --modulation_ends=50 --alpha_=0.6 > train_OGM_Shapley_MMPareto_MELD.log 2>&1 &

```

If you want to evaluate the performance on IEMOCAP dataset, you can run the `train_OGM_Shapley_MMPareto_IEMOCAP.py` file:

```bash
nohup python -u train_OGM_Shapley_MMPareto_IEMOCAP.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 8 --graph_type='hyper' --epochs=80 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L=1 --num_K=1 --modulation_ends=80 --alpha_=0.6> train_OGM_Shapley_MMPareto_IEMOCAP.log 2>&1 &
```


