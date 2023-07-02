from linevul_helpers import clean_special_token_values, get_word_att_scores

def extract_line_attention(attentions, all_tokens):
    # attentions: a tuple with of one Tensor with 4D shape (batch_size, num_heads, sequence_length, sequence_length)
    attentions = attentions[0][0]
    attention = None
    # go into the layer
    for i in range(len(attentions)):
        layer_attention = attentions[i]
        # summerize the values of each token dot other tokens
        layer_attention = sum(layer_attention)
        if attention is None:
            attention = layer_attention
        else:
            attention += layer_attention
    # clean att score for <s> and </s>
    attention = clean_special_token_values(attention, padding=True)
    # attention should be 1D tensor with seq length representing each token's attention value
    # word_att_scores -> [[token, att_value], [token, att_value], ...]
    word_att_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attention)


    # go through each line
    separator = ["Ċ", " Ċ", "ĊĊ", " ĊĊ"]
    score_sum = 0
    line = ""
    score_sum = 0
    lines_with_score = []  # line_idx, content, score
    line_idx = 0
    for i in range(len(word_att_scores)):
        score_sum += word_att_scores[i][1]
        if word_att_scores[i][0] not in separator:
            line += word_att_scores[i][0]
        else:
            lines_with_score.append((line_idx, line, score_sum.detach().item()))
            line = ""
            score_sum = 0
            line_idx += 1
    return sorted(lines_with_score, key=lambda x: x[2], reverse=True), line_idx
