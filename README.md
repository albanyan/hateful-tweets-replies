# Hateful Tweets and Replies

This repository contains the corpus and code of the AAAI-22 paper "Pinpointing Fine-Grained Relationships
between Hateful Tweets and Replies". Authors: Abdullah Albanyan and Eduardo Blanco.
<br />
[[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/21284)]
[[supplementary materials](/docs/hate-twitter-supplemental.pdf)]

## Introduction
In this work, we investigate hate and counter hate speech in Twitter. We work with hateful tweets and replies, and study the relationships between them beyond whether the reply counters the hateful tweet. In particular:

* Q1: Does the reply contain _counter hate_?
* If Q1 is _Yes_ (the reply counters the hateful tweet):
   * Q2: Provides a _justification_?
   * Q3: _Attacks the author_ of the original tweet?
* If Q1 is _No_ (the reply agrees with the hateful tweet):
   * Q4: Adds _additional hate_?



## Example

Consider the following hateful tweet which attacks Michelle Obama's physical appearance:
<p align="center">
 <kbd>
<img  src="docs/figs/tweet.png" width=50% height=50%>
  </kbd>
</p>

This reply disapproves of the hateful tweet and provides an opinion or a justification about Michelle Obama:
<p align="center">
 <kbd>
<img src="docs/figs/reply1.png" width=50% height=50%>
  </kbd>
</p>

This reply approves of the hateful tweet and uses sarcasm to introduce additional hate (not being pretty vs. being a male):
<p align="center">
  <kbd>
<img src="docs/figs/reply2.png" width=50% height=50%>
  </kbd>
</p>

<!-- ****************************************************************************************** -->
## Citation

```
@article{Albanyan_Blanco_2022,
  title = {Pinpointing Fine-Grained Relationships between Hateful Tweets and Replies},
  volume = {36},
  url = {https://ojs.aaai.org/index.php/AAAI/article/view/21284},
  doi = {10.1609/aaai.v36i10.21284},
  abstractnote = {Recent studies in the hate and counter hate domain have provided the grounds for investigating how to detect this pervasive content in social media. These studies mostly work with synthetic replies to hateful content written by annotators on demand rather than replies written by real users. We argue that working with naturally occurring replies to hateful content is key to study the problem. Building on this motivation, we create a corpus of 5,652 hateful tweets and replies. We analyze their fine-grained relationships by indicating whether the reply (a) is hate or counter hate speech, (b) provides a justification, (c) attacks the author of the tweet, and (d) adds additional hate. We also present linguistic insights into the language people use depending on these fine-grained relationships. Experimental results show improvements (a) taking into account the hateful tweet in addition to the reply and (b) pretraining with related tasks.},
  number = {10},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  author = {Albanyan, Abdullah and Blanco, Eduardo},
  year = {2022},
  month = jun,
  pages = {10418-10426},
  month_numeric = {6}
}
```
