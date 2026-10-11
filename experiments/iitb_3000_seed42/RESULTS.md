# Translation fine-tuning results

Trained on 3000 IIT Bombay pairs for 2 epochs. Checkpoint selected using 300 validation pairs.

On the same 500 held-out pairs, pretrained BLEU was 10.41 and fine-tuned BLEU was 10.04 (difference -0.37).

A positive difference indicates higher reference overlap on this sample; a zero or negative difference means fine-tuning did not improve BLEU on this sample. Possible explanations to investigate include sample size, domain mismatch, and overfitting; the scores alone do not identify the cause. Compare validation loss and individual translations.

- Small, length-filtered sample with bounded-buffer training shuffle; not representative of the whole corpus.
- General-domain IIT Bombay results do not establish privacy-policy translation quality.
- Unknown overlap with the original pretrained model training data; held out from this fine-tuning run only.
- One seed; BLEU/chrF differences do not establish statistical significance or preserved legal meaning.
