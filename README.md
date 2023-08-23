# CLIPMH
CLIP Multi-modal Hashing: A new baseline

## Abstract
Multi-modal hashing method is widely used in multimedia retrieval. It can fuse multi-source data to generate binary hash code. However, the current multi-modal methods have the problem of low retrieval accuracy. The reason is that the individual backbone networks have limited feature expression capabilities and are not jointly pre-trained on large-scale unsupervised multi-modal data. To solve this problem, we propose a new baseline CLIP Multi-modal Hashing (CLIPMH) method. It uses CLIP model to extract text and image features, and then fuse to generate hash code. CLIP improves the expressiveness of each modal feature. In this way, it can greatly improve the retrieval performance of multi-modal hashing methods. In comparison to state-of-the-art unsupervised and supervised multi-modal hashing methods, experiments reveal that the proposed CLIPMH can significantly enhance performance (Maximum increase of $8.38\%$). CLIP also has great advantages over the text and visual backbone networks commonly used before. The source codes of our CLIPMH is publicly available at: https://github.com/HackerHyper/CLIPMH.

## ARCH

