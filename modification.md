### Todo

1. the training function need to be rewrite:
    - current logic: uni-bert model for both question and answer, but we need bert_q and bert_a (dual bert style) for question and answer in the original paper
        to avoid the loss increase situation.
        want the phi_q(q) ~= phi_a(a[n]).
    - current code only did the question embedding once and not update after that
2. clean the repo and remove the useless files
3. adding the parser/unified python file that can run directly.
