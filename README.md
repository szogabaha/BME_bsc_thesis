# BME_bsc_thesis
Notebooks and dataset of the BSc thesis

## Description

Large pretrained language models have revolutionized the field of natural language processing.
Google’s BERT is the most prominent language model with a wide range of applications and a
whole family of BERT-like model derivatives. Along with the original English models, a massively
multilingual model, mBERT was released. mBERT was trained on over 100 languages including
Hungarian.

Since these models are complex non-linear functions with over a hundred million trainable
parameters, their interpretation is difficult. Particularly mBERT’s astounding cross-lingual
capabilities have gathered considerable interest. A popular evaluation and interpretation method of
black box models is probing or the use of diagnostic classifiers. Probing takes an existing model
and uses its representation of some probing sentences as the input to a small classifier that has to
classify according to some label. If the classifier is accurate, the model contains the information.
mBERT shares its representation space for 100 languages which gives rise to the question, do
related languages have similar representations in mBERT? Linguistic typology databases such as
the World Atlas of Language Structures (WALS) categorize languages according to a large number
of linguistic features. The task of the student is to find compare these typological databases with
mBERT and other multilingual model’s representations.

### Tasks to be performed by the student will include:
 Explore linguistic typology databases such as WALS.
 Train morphological probing classifiers on mBERT.
 Apply classification methods for recovering WALS’s categories.
 Apply clustering methods.
 Analyze the results.
