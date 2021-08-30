import torch
from tqdm import tqdm
from transformers import BertLMHeadModel, BertTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast
from nlp import load_dataset

if __name__ == '__main__':

    device = 'cuda'
    #model_id = 'gpt2'
    model_id = 'bert-large-cased'
    model = BertLMHeadModel.from_pretrained(model_id).to(device)
    #model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_id)
    #tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

    #max_length = model.config.n_positions
    max_length = model.config.max_position_embeddings
    stride = int(max_length/2)

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    print(f'Avg perplexity: {ppl}')