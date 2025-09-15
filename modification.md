### Todo

1. the training function need to be rewrite:
    - current logic: uni-bert model for both question and answer, but we need bert_q and bert_a (dual bert style) for question and answer in the original paper
        to avoid the loss increase situation.
        want the phi_q(q) ~= phi_a(a[n]).
    - current code only did the question embedding once and not update after that
    - training procedure:
        - dual bert setup question_bert (phi_q) and reference_bert (phi_r):
        - embed q_i and a_i,j with question_bert and reference_bert respectively
        - compute the loss and update the parameters
        - update the question_bert and reference_bert
        - repeat until the loss is stable
    - inference procedure, after we have the trained phi_q and phi_r, do:
        - embedding all chunks with phi_r.
        - embedding the new question with phi_q.
        - compute the similarity between the new question and all chunks.
        - select the chunk with the highest similarity.
        - return the selected chunk.
2. clean the repo and remove the useless files
3. adding the parser/unified python file that can run directly.
4. test file and Jupyter notebooks for explaining
5. Requirements